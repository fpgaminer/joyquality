#!/usr/bin/env python3
"""
Score all the images using our scoring model.
"""
import torch
from tqdm import tqdm
import PIL.Image
import argparse
from pathlib import Path
import psycopg
import random
import torch.nn as nn
from transformers import AutoModel
import torchvision.transforms.functional as TVF
from torch.utils.data import Dataset, DataLoader


PIL.Image.MAX_IMAGE_PIXELS = 933120000


parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=128)


@torch.inference_mode()
def main():
	args = parser.parse_args()

	# Load model
	model = So400m512(dropout=0.0)
	model.load_state_dict(torch.load("checkpoints/05k047vn/latest.pt"))
	model.requires_grad_(False)
	model.eval()
	model = model.to("cuda")
	model = torch.compile(model)

	# Fetch a list of all paths we need to work with
	with psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd().parent / "pg-socket")) as conn, conn.cursor('score-images') as cur:
		cur.execute("SELECT path FROM images WHERE embedding IS NOT NULL AND bt_score IS NULL")
		paths = [path for path, in tqdm(cur, desc="Fetching paths from database...", dynamic_ncols=True)]
	
	print(f"Found {len(paths)} paths to process")
	random.shuffle(paths)

	# Create dataloader
	dataset = ImageDataset(paths)
	dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=32, pin_memory=True, drop_last=False, shuffle=False)

	# Score images in batches and update DB
	with psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd().parent / "pg-socket")) as conn, conn.cursor() as cur, tqdm(total=len(paths), desc="Scoring images...", dynamic_ncols=True) as pbar:
		for i, pixel_values in enumerate(dataloader):
			# Run through the model
			with torch.amp.autocast("cuda", dtype=torch.bfloat16): # type: ignore
				scores = model(pixel_values.to("cuda", non_blocking=True))
			
			scores = scores.detach().float().cpu().tolist()
			cur.executemany("UPDATE images SET bt_score = %s WHERE path = %s", 
				[(score, paths[i * args.batch_size + j]) for j, score in enumerate(scores)]
			)
			conn.commit()
			pbar.update(len(scores))


class So400m512(nn.Module):
	def __init__(self, dropout: float):
		super(So400m512, self).__init__()

		self.clip_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-512", dtype=torch.float32)
		self.clip_model = self.clip_model.vision_model
		self.head = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(self.clip_model.config.hidden_size, 1, bias=True),
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		embedding = self.clip_model(x).pooler_output
		return self.head(embedding).squeeze(-1)


class ImageDataset(Dataset):
	def __init__(self, paths: list[str | Path]):
		self.paths = paths
	
	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, idx: int):
		image = PIL.Image.open(self.paths[idx])
		image = image.convert("RGB")
		if image.size != (512, 512):
			image = image.resize((512, 512), PIL.Image.Resampling.BICUBIC)  # Model was trained with bicubic resizing, which performed better than Lanczos
		
		new_image = PIL.Image.new("RGB", (512, 512), (128, 128, 128)) # type: ignore
		new_image.paste(image, (0, 0))

		pixel_values = TVF.pil_to_tensor(new_image) / 255.0       # [0, 1]
		pixel_values = TVF.normalize(pixel_values, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]

		return pixel_values


if __name__ == "__main__":
	main()
