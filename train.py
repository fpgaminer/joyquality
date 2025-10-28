#!/usr/bin/env python3
import numpy as np
import random
from pathlib import Path
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import get_scheduler
from torch import optim
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
import io
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoModel, Owlv2VisionModel, DINOv3ViTModel, CLIPVisionModel
import torchvision.transforms.functional as TVF
import json
import wandb
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import math
import types, inspect
import hashlib
import importlib.metadata as md
import functools
import sys
import tempfile
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from functools import partial


Image.MAX_IMAGE_PIXELS = 300_000_000


@dataclass
class TrainerConfig:
	seed: int = 69
	wandb_project: str = "quality-arena-full-1"
	batch_size: int = 256
	device_batch_size: int = 16
	total_samples: int = 10000   # Total samples to train on
	optimizer: str = "adamw"
	learning_rate: float = 8e-5
	min_learning_rate: float = 0.0  # minimum fraction of the peak LR to keep after decay
	adam_beta1: float = 0.9
	adam_beta2: float = 0.999
	adam_eps: float = 1e-8
	weight_decay: float = 0.01
	dropout: float = 0.0
	clip_grad_norm: float = 1.0
	lr_scheduler: str = "onecycle"   # onecycle, cosine, linear
	warmup_samples: int = 1000       # Number of samples to warmup LR over (if using cosine or linear)
	test_every: int = 2000
	compile_model: bool = True
	model: str = "FullQualityModel"
	checkpoint_path: Path = Path("checkpoints")
	dataset: Path = Path("pairs-dataset-2.json")
	base_checkpoint: Path | None = None
	full_train: bool = False          # If False, only the head of the model is trained
	cache_images: bool = False

cs = ConfigStore.instance()
cs.store(name="trainer_config", node=TrainerConfig)
cs.store(
	group="hydra",       # overrides Hydra’s own defaults
	name="no_io",        # arbitrary name
	node=OmegaConf.create({
		"run": {"dir": "."},        # stay in the original CWD
		"output_subdir": None,      # don’t copy configs to .hydra/
		"job": {"chdir": False},    # don’t chdir into a run dir
		# turn off every file‑logging handler
		"job_logging": "disabled",  # built‑in config: no log file for your job
		"hydra_logging": "disabled" # disable Hydra’s internal log file
	}),
)



class Trainer:
	def __init__(self, config: TrainerConfig):
		self.config = config
		self.logger = logging.getLogger(__name__)
		logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		self.logger.setLevel(logging.INFO)

		# Performance enhancing drugs
		torch.set_float32_matmul_precision("high")
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True

		random.seed(config.seed)
		np.random.seed(config.seed)
		torch.manual_seed(config.seed)
		torch.cuda.manual_seed(config.seed)

		# Calculate device batch size and such
		self.device_batch_size = min(config.batch_size, config.device_batch_size)
		self.gradient_accumulation_steps = config.batch_size // self.device_batch_size
		self.total_steps = config.total_samples // config.batch_size
		self.total_device_batches = self.total_steps * self.gradient_accumulation_steps
		self.test_every_step = config.test_every // config.batch_size
		assert config.batch_size == self.device_batch_size * self.gradient_accumulation_steps, f"Batch size {config.batch_size} must be divisible by device batch size {self.device_batch_size} for gradient accumulation steps {self.gradient_accumulation_steps}"

		self.min_lr_fraction = float(config.min_learning_rate)
		if not 0.0 <= self.min_lr_fraction <= 1.0:
			raise ValueError(f"min_learning_rate must be between 0 and 1 inclusive, got {self.min_lr_fraction}")

		# Build model
		cls = globals()[config.model]
		self.model = cls(dropout=config.dropout, full_train=config.full_train).to('cuda')

		if config.base_checkpoint is not None:
			self.logger.info(f"Loading base checkpoint from {config.base_checkpoint}")
			state_dict = torch.load(config.base_checkpoint, map_location='cpu')
			self.model.load_state_dict(state_dict)

		self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
		self.logger.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters, of which {sum(p.numel() for p in self.trainable_params):,} are trainable.")
		if config.compile_model:
			self.model = torch.compile(self.model)  # type: ignore

		# Build optimizer
		if config.optimizer == "adamw":
			self.optimizer = optim.AdamW(
				self.trainable_params,
				lr=config.learning_rate,
				betas=(config.adam_beta1, config.adam_beta2),
				eps=config.adam_eps,
				weight_decay=config.weight_decay,
			)
		elif config.optimizer == "lion":
			from lion_pytorch import Lion # type: ignore
			self.optimizer = Lion(
				self.trainable_params,
				lr=config.learning_rate,
				betas=(config.adam_beta1, config.adam_beta2),
				weight_decay=config.weight_decay,
			)
		else:
			raise ValueError(f"Unknown optimizer: {config.optimizer}")
		
		# Build LR scheduler
		num_warmup_steps = config.warmup_samples // config.batch_size
		if self.config.lr_scheduler == "onecycle":
			onecycle_kwargs = {
				"optimizer": self.optimizer,
				"max_lr": config.learning_rate,
				"total_steps": self.total_steps,
			}
			if self.min_lr_fraction > 0.0:
				onecycle_kwargs["final_div_factor"] = 1.0 / self.min_lr_fraction
			self.lr_scheduler = OneCycleLR(**onecycle_kwargs)
		elif self.config.lr_scheduler == "cosine":
			self.lr_scheduler = get_cosine_schedule_with_warmup(
				optimizer=self.optimizer,
				num_warmup_steps=num_warmup_steps,
				num_training_steps=self.total_steps,
				min_lr_ratio=self.config.min_learning_rate,
			)
		elif self.config.lr_scheduler == "linear":
			self.lr_scheduler = get_scheduler(
				name="linear",
				optimizer=self.optimizer,
				num_warmup_steps=num_warmup_steps,
				num_training_steps=self.total_steps,
			)
		else:
			raise ValueError(f"Unknown LR scheduler: {config.lr_scheduler}")
		
		# Build dataset
		train_dataset, test_dataset, dist_mx = build_datasets(self.config.dataset, self.model.get_preprocessor(), cache_images=self.config.cache_images)
		self.dist_mx = dist_mx

		# Load ranked validation set if available
		ranked_validation_path = self.config.dataset.with_suffix(".ranked.json")

		if ranked_validation_path.exists():
			_, validation_ranks_bt, validation_ranks_paths, _ = load_validation_ranking("pretraining_ranked_validation_results.json")
			self.validation_ranks_bt = validation_ranks_bt
			self.validation_ranks_paths = validation_ranks_paths
			self.validation_ranks_dataset = SingleImages(validation_ranks_paths, self.model.get_preprocessor())
			self.validation_ranks_inputs = torch.stack([self.validation_ranks_dataset[i] for i in range(len(self.validation_ranks_dataset))], dim=0)
			self.logger.info(f"Validation Ranks Inputs shape: {self.validation_ranks_inputs.shape}")
		else:
			self.validation_ranks_bt = None
			self.validation_ranks_paths = None
			self.validation_ranks_dataset = None
			self.validation_ranks_inputs = None

		validation_paths = list(set([p for p, _, _, _ in test_dataset.pairs] + [p for _, p, _, _ in test_dataset.pairs]))
		jpeg_validation_dataset = CorruptPairwiseImages(validation_paths, self.model.get_preprocessor(), jpeg_compression_augment)
		dl = DataLoader(dataset=jpeg_validation_dataset, batch_size=self.device_batch_size, shuffle=False, drop_last=False, num_workers=32)
		self.jpeg_validation_inputs = []
		for batch in tqdm(dl, desc="Building JPEG validation inputs", dynamic_ncols=True):
			for i in range(batch[0].shape[0]):
				self.jpeg_validation_inputs.append((batch[0][i], batch[1][i], batch[2][i]))
		#self.jpeg_validation_inputs = [self.jpeg_validation_dataset[i] for i in tqdm(range(len(self.jpeg_validation_dataset)), desc="Building JPEG validation inputs", dynamic_ncols=True)]

		lowres_validation_dataset = CorruptPairwiseImages(validation_paths, self.model.get_preprocessor(), lowres_augment)
		dl = DataLoader(dataset=lowres_validation_dataset, batch_size=self.device_batch_size, shuffle=False, drop_last=False, num_workers=32)
		self.lowres_validation_inputs = []
		for batch in tqdm(dl, desc="Building low-res validation inputs", dynamic_ncols=True):
			for i in range(batch[0].shape[0]):
				self.lowres_validation_inputs.append((batch[0][i], batch[1][i], batch[2][i]))
		#self.lowres_validation_inputs = [self.lowres_validation_dataset[i] for i in tqdm(range(len(self.lowres_validation_dataset)), desc="Building low-res validation inputs", dynamic_ncols=True)]

		# Build dataloaders
		self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.device_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=32, persistent_workers=True)
		self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.device_batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=32, persistent_workers=True)

	def fit(self):
		device_step = 0
		self.global_step = 0
		self.global_samples_seen = 0

		self.logger.info("Starting training...")
		wandb_config = dict(self.config) # type: ignore
		wandb.init(project=self.config.wandb_project, config=wandb_config)
		wandb.watch(self.model, log_freq=20) # type: ignore

		# Test saving so we can crash early
		self.save_model()

		# Initial test without any training
		results = self.run_test()
		wandb.log({
			"test/accuracy": results['accuracy'],
			"test/nll": results['nll_mean'],
			"test/mcfadden_R2_vs_0p5": results['mcfadden_R2_vs_0p5'],
			"test/auc": results['auc'],
			"test/brier": results['brier'],
			"test/ece": results['ece'],
			"test/dec_median_abs_p_minus_0p5": results['dec_median_abs_p_minus_0p5'],
			"test/dec_strong_share_|p-0.5|>0.2": results['dec_strong_share_|p-0.5|>0.2'],
			"test/aurc": results['aurc'],
			"test/weighted/auc": results['weighted']['auc'] if results['weighted'] is not None else float('nan'),
			"test/weighted/brier": results['weighted']['brier'] if results['weighted'] is not None else float('nan'),
			"test/weighted/ece": results['weighted']['ece'] if results['weighted'] is not None else float('nan'),
			"test/weighted/weight_mean": results['weighted']['weight_mean'] if results['weighted'] is not None else float('nan'),
			"test/weighted/weight_std": results['weighted']['weight_std'] if results['weighted'] is not None else float('nan'),
			"test/rank/kendall_tau": results['rank_kendall_tau'] if results['rank_kendall_tau'] is not None else float('nan'),
			"test/rank/spearman_rho": results['rank_spearman_rho'] if results['rank_spearman_rho'] is not None else float('nan'),
			"test/jpeg_validation_accuracy": results['jpeg_validation_accuracy'],
			"test/lowres_validation_accuracy": results['lowres_validation_accuracy'],
			"test/samples": 0,
		}, step=0)

		loss_sum = torch.tensor(0.0, device='cuda', requires_grad=False, dtype=torch.float32)
		dataloader_iter = iter(self.train_dataloader)
		pbar = tqdm(total=self.total_device_batches * self.device_batch_size, desc="Training", dynamic_ncols=True, smoothing=0.01)
		with logging_redirect_tqdm():
			for device_step in range(0, self.total_device_batches):
				self.global_step = device_step // self.gradient_accumulation_steps
				self.global_samples_seen = (device_step + 1) * self.device_batch_size

				self.model.train()

				# Get the next batch
				try:
					batch, y, _ = next(dataloader_iter)
				except StopIteration:
					self.logger.info("Dataloader exhausted, starting new epoch")
					dataloader_iter = iter(self.train_dataloader)
					batch, y, _ = next(dataloader_iter)
				
				# Move to device
				batch = batch.to('cuda', non_blocking=True).to(torch.float32)
				y = y.to('cuda', non_blocking=True)

				is_last_device_step = (device_step + 1) % self.gradient_accumulation_steps == 0
				is_last_step = (self.global_step + 1) == self.total_steps

				with torch.amp.autocast("cuda", dtype=torch.bfloat16): # type: ignore
					# Forward pass
					pixel_values = batch.flatten(0, 1)  # [2*B, ...]
					scores = self.model(pixel_values)    # [2*B]
					scores = scores.unflatten(0, (batch.shape[0], 2))  # [B, 2]
					sL = scores[:,0]
					sR = scores[:,1]
				
					loss = bt_loss(sL, sR, None)
					loss = loss / self.gradient_accumulation_steps
					loss_sum.add_(loss.detach())

				# Backward pass
				loss.backward()

				if is_last_device_step:
					pbar.set_description(f"Loss: {loss_sum.item():.4f}")

					# Clip gradients
					torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.clip_grad_norm)

					# Take a step
					self.optimizer.step()
					self.lr_scheduler.step()
					self.optimizer.zero_grad(set_to_none=True)

					wandb.log({
						"train/samples": self.global_samples_seen,
						"train/loss": loss_sum.item(),
						"train/lr": self.lr_scheduler.get_last_lr()[0],
					}, step=self.global_step)
					loss_sum.zero_()
				
					# Test
					if (self.global_step + 1) % self.test_every_step == 0 or is_last_step:
						results = self.run_test()
						wandb.log({
							"test/accuracy": results['accuracy'],
							"test/nll": results['nll_mean'],
							"test/mcfadden_R2_vs_0p5": results['mcfadden_R2_vs_0p5'],
							"test/auc": results['auc'],
							"test/brier": results['brier'],
							"test/ece": results['ece'],
							"test/dec_median_abs_p_minus_0p5": results['dec_median_abs_p_minus_0p5'],
							"test/dec_strong_share_|p-0.5|>0.2": results['dec_strong_share_|p-0.5|>0.2'],
							"test/aurc": results['aurc'],
							"test/weighted/auc": results['weighted']['auc'] if results['weighted'] is not None else float('nan'),
							"test/weighted/brier": results['weighted']['brier'] if results['weighted'] is not None else float('nan'),
							"test/weighted/ece": results['weighted']['ece'] if results['weighted'] is not None else float('nan'),
							"test/weighted/weight_mean": results['weighted']['weight_mean'] if results['weighted'] is not None else float('nan'),
							"test/weighted/weight_std": results['weighted']['weight_std'] if results['weighted'] is not None else float('nan'),
							"test/rank/kendall_tau": results['rank_kendall_tau'] if results['rank_kendall_tau'] is not None else float('nan'),
							"test/rank/spearman_rho": results['rank_spearman_rho'] if results['rank_spearman_rho'] is not None else float('nan'),
							"test/jpeg_validation_accuracy": results['jpeg_validation_accuracy'],
							"test/lowres_validation_accuracy": results['lowres_validation_accuracy'],
							"test/samples": self.global_samples_seen,
						}, step=self.global_step)
						if results['per_distance_bins'] is not None:
							self.logger.info(f"Binned test results at step {self.global_step}:")
							for bin_info in results['per_distance_bins']['bins']:
								self.logger.info(f"  Distance bin {bin_info['bin']}: n={bin_info['n']}, acc={bin_info.get('acc', float('nan'))}, auc={bin_info.get('auc', float('nan'))}, brier={bin_info.get('brier', float('nan'))}, ece={bin_info.get('ece', float('nan'))}, dec_median={bin_info.get('dec_median', float('nan'))}, dec_strong_>0.2={bin_info.get('dec_strong_>0.2', float('nan'))}")
				
				if is_last_step:
					self.save_model()

				pbar.update(self.device_batch_size)
			
			pbar.close()
		
		self.logger.info("Training complete.")
	
	@torch.no_grad()
	def run_test(self, ece_bins: int = 15, dist_bins: int = 10) -> dict:
		self.model.eval()

		nll_sum = torch.zeros(1, dtype=torch.float32, device='cuda', requires_grad=False)
		correct = 0
		total = 0

		probs_left = []
		labels = []
		distances = []

		for batch, y, d in tqdm(self.test_dataloader, desc="Testing", dynamic_ncols=True, leave=False):
			batch = batch.to('cuda').to(torch.float32) # [B, 2, D]
			y = y.to('cuda')   # [B]
			with torch.amp.autocast('cuda', dtype=torch.bfloat16): # type: ignore
				pixel_values = batch.flatten(0, 1)  # [2*B, ...]
				scores = self.model(pixel_values)  # [2*B]
				scores = scores.unflatten(0, (batch.shape[0], 2))  # [B, 2]
				sL = scores[:,0]
				sR = scores[:,1]
				loss = bt_loss(sL, sR, None)
				p_left = torch.sigmoid(sL - sR)
			
			bsz = y.shape[0]
			nll_sum.add_(loss * bsz)
			total += bsz
			correct += ((sL > sR).to(torch.long)).sum().item()

			probs_left.append(p_left.detach().float().cpu().numpy())
			labels.append(y.detach().int().cpu().numpy())
			distances.append(d.detach().float().cpu().numpy())
		
		p = np.concatenate(probs_left, axis=0)
		ytrue = np.concatenate(labels, axis=0)

		nll_mean = float((nll_sum / total).item())
		accuracy = float(((p >= 0.5) == (ytrue == 1)).mean())
		auc = _auc(ytrue, p)
		brier = _brier(p, ytrue)
		ece = _ece(p, ytrue, n_bins=ece_bins)
		dec_med = float(np.median(np.abs(p - 0.5)))
		dec_strong = float(np.mean(np.abs(p - 0.5) > 0.2))
		aurc_val = _aurc(p, ytrue)
		r2_mcf = _r2_mcfadden(nll_mean)

		if self.dist_mx is not None:
			d = np.concatenate(distances, axis=0)

			# per-distance slices
			dist_edges = _make_quantile_bins(d, n_bins=dist_bins)
			per_distance_bins = {
				'edges': dist_edges.tolist(),
				'bins': _per_distance_bins(p, ytrue, d, dist_edges, ece_bins=ece_bins),
			}

			# importance-weighted metrics (proxy real-world)
			weighted = None
			w, _ = _importance_weights(d, self.dist_mx, n_bins=20)
			weighted = {
				'auc': _auc(ytrue, p, w=w),
				'brier': _brier(p, ytrue, w=w),
				'ece': _ece(p, ytrue, n_bins=ece_bins, w=w),
				'weight_mean': float(np.mean(w)),
				'weight_std': float(np.std(w)),
			}
		else:
			per_distance_bins = None
			weighted = None

		if self.validation_ranks_paths is not None and self.validation_ranks_bt is not None and self.validation_ranks_inputs is not None:
			# Compare ranked validation against ground truth ranking
			model_scores = []
			for i in range(0, len(self.validation_ranks_paths), self.device_batch_size):
				batch = self.validation_ranks_inputs[i:i+self.device_batch_size].to('cuda').to(torch.float32)
				with torch.amp.autocast('cuda', dtype=torch.bfloat16): # type: ignore
					scores = self.model(batch).detach().float().cpu().numpy()
					model_scores.extend(scores.tolist())
			
			model_scores = {fh: score for fh, score in zip(self.validation_ranks_bt, model_scores)}
			order_model = sorted(model_scores.keys(), key=lambda k: model_scores[k], reverse=True) # type: ignore

			# Kendall τ
			tau = kendall_tau_fast(self.validation_ranks_bt, order_model)

			# Spearman ρ
			pos_h = {k: i for i, k in enumerate(self.validation_ranks_bt)}
			pos_m = {k: i for i, k in enumerate(order_model)}
			n = len(self.validation_ranks_bt)
			ss = sum((pos_h[k] - pos_m[k])**2 for k in self.validation_ranks_bt)
			rho = 1.0 - 6.0*ss/(n*(n**2 - 1)) if n > 1 else 1.0

			self.logger.info(f"Model Validation Ranking: {', '.join([f'{k.hex()}: {model_scores[k]:.4f}' for k in order_model])}")
		else:
			tau = None
			rho = None

		# JPEG evaluation
		model_scores = []
		for i in range(0, len(self.jpeg_validation_inputs), self.device_batch_size):
			batch = self.jpeg_validation_inputs[i:i+self.device_batch_size]
			pixel_values = torch.stack([x[0] for x in batch], dim=0).to('cuda').to(torch.float32)
			valid = [x[2] for x in batch]

			with torch.amp.autocast('cuda', dtype=torch.bfloat16): # type: ignore
				pixel_values = pixel_values.flatten(0, 1)  # [2*B, ...]
				scores = self.model(pixel_values)  # [2*B]
				scores = scores.unflatten(0, (len(batch), 2))  # [B, 2]
				sL = scores[:,0]
				sR = scores[:,1]
				scores = (sL - sR).detach().float().cpu().numpy()
				for score, v in zip(scores, valid):
					if v:
						model_scores.append(score)
		jpeg_accuracy = float((np.array(model_scores) > 0).mean())

		# Low-res evaluation
		model_scores = []
		for i in range(0, len(self.lowres_validation_inputs), self.device_batch_size):
			batch = self.lowres_validation_inputs[i:i+self.device_batch_size]
			pixel_values = torch.stack([x[0] for x in batch], dim=0).to('cuda').to(torch.float32)
			valid = [x[2] for x in batch]

			with torch.amp.autocast('cuda', dtype=torch.bfloat16): # type: ignore
				pixel_values = pixel_values.flatten(0, 1)  # [2*B, ...]
				scores = self.model(pixel_values)  # [2*B]
				scores = scores.unflatten(0, (len(batch), 2))  # [B, 2]
				sL = scores[:,0]
				sR = scores[:,1]
				scores = (sL - sR).detach().float().cpu().numpy()
				for score, v in zip(scores, valid):
					if v:
						model_scores.append(score)
		lowres_accuracy = float((np.array(model_scores) > 0).mean())

		return {
			'n': int(len(p)),
			'accuracy': accuracy,
			'nll_mean': nll_mean,
			'mcfadden_R2_vs_0p5': r2_mcf,
			'auc': auc,
			'brier': brier,
			'ece': ece,
			'dec_median_abs_p_minus_0p5': dec_med,
			'dec_strong_share_|p-0.5|>0.2': dec_strong,
			'aurc': aurc_val,
			'per_distance_bins': per_distance_bins,
			'weighted': weighted,
			'rank_kendall_tau': tau,
			'rank_spearman_rho': rho,
			'jpeg_validation_accuracy': jpeg_accuracy,
			'lowres_validation_accuracy': lowres_accuracy,
		}
	
	def save_model(self):
		assert wandb.run is not None, "Wandb run is not initialized"
		path = self.config.checkpoint_path / wandb.run.id / "latest.pt"
		path.parent.mkdir(parents=True, exist_ok=True)

		# Unwrap compiled model for saving
		if isinstance(self.model, torch._dynamo.eval_frame.OptimizedModule):
			model = self.model._orig_mod  # type: ignore
		else:
			model = self.model

		torch.save(model.state_dict(), path)


#### Metric functions ####
def _brier(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
	if w is None:
		return float(np.mean((p - y) ** 2))
	else:
		return float(np.average((p - y) ** 2, weights=w))


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15, w: np.ndarray | None = None) -> float:
	conf = np.maximum(p, 1.0 - p)
	bins = np.linspace(0.0, 1.0, n_bins + 1)
	idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
	N = len(p)
	ece = 0.0
	for b in range(n_bins):
		m = (idx == b)
		if not np.any(m): continue
		if w is None:
			wb = m.sum() / max(N, 1)
			acc_b = float(((p[m] >= 0.5) == (y[m] == 1)).mean())
			conf_b = float(conf[m].mean())
		else:
			wb = float(w[m].sum() / (w.sum() + 1e-12))
			acc_b = float(np.average(((p[m] >= 0.5) == (y[m] == 1)).astype(float), weights=w[m]))
			conf_b = float(np.average(conf[m], weights=w[m]))
		ece += wb * abs(acc_b - conf_b)
	return float(ece)


def _auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
	y = y.astype(np.int32); s = s.astype(np.float64)
	if w is None: w = np.ones_like(s, dtype=np.float64)
	Wp, Wn = w[y == 1].sum(), w[y == 0].sum()
	if Wp <= 0 or Wn <= 0: return float('nan')
	order = np.argsort(s, kind='mergesort')
	y_ord, s_ord, w_ord = y[order], s[order], w[order]
	uniq, start = np.unique(s_ord, return_index=True)
	end = np.r_[start[1:], len(s_ord)]
	cum_neg = 0.0; num = 0.0
	for a, b in zip(start, end):
		w_pos = w_ord[a:b][y_ord[a:b] == 1].sum()
		w_neg = w_ord[a:b][y_ord[a:b] == 0].sum()
		num += w_pos * (cum_neg + 0.5 * w_neg)
		cum_neg += w_neg
	return float(num / (Wp * Wn))


def _aurc(p: np.ndarray, y: np.ndarray) -> float:
	conf = np.abs(p - 0.5)
	order = np.argsort(-conf)
	pred = (p[order] >= 0.5).astype(np.int32)
	yord = y[order].astype(np.int32)
	err = (pred != yord).astype(np.float64)
	cum_err = np.cumsum(err)
	k = np.arange(1, len(err) + 1)
	risk = cum_err / k
	cov = k / len(err)
	return float(np.trapz(risk, cov))


def _r2_mcfadden(nll_mean: float) -> float:
	return float(1.0 - (nll_mean / (math.log(2.0) + 1e-12)))


def _make_quantile_bins(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
	qs = np.linspace(0, 1, n_bins + 1)
	edges = np.unique(np.quantile(values, qs))
	if len(edges) < 2:
		edges = np.array([values.min(), values.max()])
	return edges


def _per_distance_bins(p: np.ndarray, y: np.ndarray, d: np.ndarray, edges: np.ndarray, ece_bins: int = 15):
	out = []
	last_hi = edges[-1]
	for lo, hi in zip(edges[:-1], edges[1:]):
		m = (d >= lo) & ((d < hi) if hi != last_hi else (d <= hi))
		if not np.any(m):
			out.append({'bin': (float(lo), float(hi)), 'n': 0})
			continue
		pm, ym = p[m], y[m]
		out.append({
			'bin': (float(lo), float(hi)),
			'n': int(m.sum()),
			'acc': float(((pm >= 0.5) == (ym == 1)).mean()),
			'auc': _auc(ym, pm),                 # NaN if single class in this bin
			'brier': _brier(pm, ym),
			'ece': _ece(pm, ym, n_bins=ece_bins),
			'dec_median': float(np.median(np.abs(pm - 0.5))),
			'dec_strong_>0.2': float(np.mean(np.abs(pm - 0.5) > 0.2)),
		})
	return out


def _importance_weights(pair_dists: np.ndarray, dist_mx: np.ndarray, n_bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
	"""Build q(d) from ALL unordered pairs in the matrix; return per-example weights normalized to mean 1."""
	iu = np.triu_indices(dist_mx.shape[0], k=1)
	all_vals = np.asarray(dist_mx[iu], dtype=np.float64)

	lo, hi = float(all_vals.min()), float(all_vals.max())
	edges = np.linspace(lo, hi, n_bins + 1)
	q_cnt, _ = np.histogram(all_vals, bins=edges)
	p_cnt, _ = np.histogram(pair_dists, bins=edges)

	q = q_cnt.astype(np.float64) + 1e-9
	p = p_cnt.astype(np.float64) + 1e-9

	idx = np.clip(np.digitize(pair_dists, edges) - 1, 0, len(edges) - 2)
	w = q[idx] / p[idx]
	w = w * (len(w) / (w.sum() + 1e-12))   # normalize to mean 1
	return w.astype(np.float64), edges


def read_dataset(path: Path) -> tuple[dict[str, list[tuple[str, str, int, float]]], np.ndarray | None]:
	npz_path = path.with_suffix(".npz")

	data = json.loads(path.read_text())

	if npz_path.exists():
		npz = np.load(npz_path, allow_pickle=True, mmap_mode='r')
		sim_mx = np.array(npz['distances'])
		dist_mx = 1.0 - sim_mx
		path_to_idx = {path: idx for idx, path in enumerate(npz['paths'])}
		for i in range(len(data["test"])):
			path1, path2, label = data["test"][i]
			idx1 = path_to_idx[path1]
			idx2 = path_to_idx[path2]
			distance = dist_mx[idx1, idx2]
			# Append distance to the tuple
			data["test"][i] = (path1, path2, label, float(distance))
	else:
		dist_mx = None
		for i in range(len(data["test"])):
			data["test"][i] = (*data["test"][i], 0.0)  # Dummy distance for test set
	
	for i in range(len(data["train"])):
		data["train"][i] = (*data["train"][i], 0.0)  # Dummy distance for train set

	return data, dist_mx


def build_datasets(path: Path, processor, cache_images: bool) -> tuple["PairwiseImages", "PairwiseImages", np.ndarray | None]:
	data, dist_mx = read_dataset(path)
	train_data = data['train']
	test_data = data['test']

	train_dataset = PairwiseImages(train_data, processor, cache_images=cache_images, cache_dir=Path("training-cache-images"))
	test_dataset = PairwiseImages(test_data, processor, cache_images=cache_images, cache_dir=Path("training-cache-images"))

	return train_dataset, test_dataset, dist_mx


def load_validation_ranking(path: str) -> tuple[dict[bytes, float], list[bytes], list[str], dict]:
	"""
	Load BT ground-truth from JSON produced by build_ground_truth().
	Returns:
		human_bt_scores: {filehash_bytes: bt_score}
		order_bt: [filehash_bytes, ...] best -> worst
		meta: the meta dict from the file (n_items, n_pairs, training_neglogloss, etc.)
	"""
	with open(path, "r") as f:
		data = json.load(f)
	ranking = data["ranking"]
	meta = data.get("meta", {})

	human_bt_scores = {bytes.fromhex(r["filehash_hex"]): float(r["bt_score"]) for r in ranking}
	order_bt = [bytes.fromhex(r["filehash_hex"]) for r in ranking]  # best -> worst
	order_bt_paths = [r["path"] for r in ranking]
	return human_bt_scores, order_bt, order_bt_paths, meta


def kendall_tau_fast(order_a: list[bytes], order_b: list[bytes]) -> float:
	"""
	O(n log n) Kendall's tau for two full orders of the same items (no ties).
	Implemented via inversion counting with a Fenwick tree.
	Returns in [-1, 1].
	"""
	assert len(order_a) == len(order_b)
	n = len(order_a)
	if n <= 1:
		return 1.0
	pos_a = {k: i for i, k in enumerate(order_a)}
	seq_b = [pos_a[k] for k in order_b]

	# Fenwick tree
	size = n + 2
	bit = [0] * (size)

	def add(i: int):
		i += 1
		while i < size:
			bit[i] += 1
			i += i & -i

	def prefix_sum(i: int) -> int:
		s = 0
		i += 1
		while i > 0:
			s += bit[i]
			i -= i & -i
		return s

	inversions = 0
	for i, v in enumerate(seq_b):
		inversions += i - prefix_sum(v)
		add(v)

	total_pairs = n * (n - 1) // 2
	return 1.0 - 2.0 * inversions / total_pairs


def bt_loss(win_scores: torch.Tensor, lose_scores: torch.Tensor, tau: nn.Parameter | None) -> torch.Tensor:
	"""
	Bradley-Terry pairwise loss.
	"""
	margin = win_scores - lose_scores
	if tau is not None:
		margin = margin / tau
	
	loss = F.softplus(-margin).mean()

	return loss


def so400_processor(images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
	image = images.convert("RGB")
	if image.size != (384, 384):
		image = image.resize((384, 384), Image.Resampling.LANCZOS)
	
	new_image = Image.new("RGB", (384, 384), (128, 128, 128)) # type: ignore
	new_image.paste(image, (0, 0))

	pixel_values = TVF.pil_to_tensor(new_image) / 255.0           # [0, 1]
	pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])  # [-1, 1]

	return {"pixel_values": pixel_values.unsqueeze(0)}


class FullQualityModel(nn.Module):
	def __init__(self, dropout: float):
		super(FullQualityModel, self).__init__()

		self.clip_model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384", torch_dtype=torch.float32)
		self.clip_model = self.clip_model.vision_model
		self.head = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(self.clip_model.config.hidden_size, 1, bias=True),
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		embedding = self.clip_model(x).pooler_output
		return self.head(embedding).squeeze(-1)
	
	def get_preprocessor(self):
		return so400_processor


class OwlModel(nn.Module):
	owl_model: Owlv2VisionModel

	def __init__(self, dropout: float):
		super(OwlModel, self).__init__()

		self.owl_model = Owlv2VisionModel.from_pretrained("google/owlv2-base-patch16-ensemble", torch_dtype=torch.float32)
		self.head = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(self.owl_model.config.hidden_size, 1, bias=True),
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		embedding = self.owl_model(x).pooler_output
		return self.head(embedding).squeeze(-1)
	
	def get_preprocessor(self):
		def processor(images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
			big_side = max(images.size)
			new_image = Image.new("RGB", (big_side, big_side), (128, 128, 128)) # type: ignore
			new_image.paste(images.convert("RGB"), (0, 0))
			preped = new_image.resize((960, 960), Image.Resampling.LANCZOS)
			pixel_values = TVF.pil_to_tensor(preped) / 255.0
			pixel_values = TVF.normalize(pixel_values, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

			return {"pixel_values": pixel_values.unsqueeze(0)}
		return processor


class DinoModel(nn.Module):
	dino_model: DINOv3ViTModel

	def __init__(self, dropout: float):
		super(DinoModel, self).__init__()

		self.dino_model = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m", torch_dtype=torch.float32)
		self.head = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(self.dino_model.config.hidden_size, 1, bias=True),
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		embedding = self.dino_model(x).pooler_output
		return self.head(embedding).squeeze(-1)
	
	def get_preprocessor(self):
		def processor(images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
			image = images.convert("RGB")
			img = image.resize((1024, 1024), Image.Resampling.LANCZOS)
			mean = torch.tensor([0.485, 0.456, 0.406])
			std = torch.tensor([0.229, 0.224, 0.225])
			img = TVF.pil_to_tensor(img) / 255.0
			img = (img - mean[:, None, None]) / std[:, None, None]

			return {"pixel_values": img.unsqueeze(0)}
		return processor


class RescaledSo400(nn.Module):
	def __init__(self, dropout: float):
		super(RescaledSo400, self).__init__()

		self.clip_model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384", torch_dtype=torch.float32)
		self.clip_model = self.clip_model.vision_model

		##### OLD: Interpolate position embeddings to 1036x1036 (74x74 patches = 5476)
		# Interpolate position embeddings to 532x532 (38x38 patches = 1444)
		with torch.no_grad():
			dtype = self.clip_model.embeddings.position_embedding.weight.dtype
			#interpolated_embeddings = self.clip_model.embeddings.interpolate_pos_encoding(torch.zeros(1, 5476, 1152, dtype=dtype), 1036, 1036)
			interpolated_embeddings = self.clip_model.embeddings.interpolate_pos_encoding(torch.zeros(1, 1444, 1152, dtype=dtype), 532, 532)
			logging.info(f"Rescaled position embeddings from {self.clip_model.embeddings.position_embedding.weight.shape} to {interpolated_embeddings.shape}")
			new_pos_emb = nn.Embedding(interpolated_embeddings.shape[1], self.clip_model.config.hidden_size).to(dtype=dtype)
			new_pos_emb.weight.copy_(interpolated_embeddings.squeeze(0))
		self.clip_model.embeddings.position_embedding = new_pos_emb
		self.clip_model.embeddings.register_buffer("position_ids", torch.arange(interpolated_embeddings.shape[1]).expand((1, -1)), persistent=False)
		self.resolution = 532

		self.head = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(self.clip_model.config.hidden_size, 1, bias=True),
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		embedding = self.clip_model(x).pooler_output
		return self.head(embedding).squeeze(-1)
	
	def get_preprocessor(self):
		def processor(images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
			image = images.convert("RGB")
			if image.size != (self.resolution, self.resolution):
				image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
			
			new_image = Image.new("RGB", (self.resolution, self.resolution), (128, 128, 128)) # type: ignore
			new_image.paste(image, (0, 0))

			pixel_values = TVF.pil_to_tensor(new_image) / 255.0           # [0, 1]
			pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])  # [-1, 1]

			return {"pixel_values": pixel_values.unsqueeze(0)}
		return processor


class So400BigPatch(nn.Module):
	def __init__(self, dropout: float):
		super(So400BigPatch, self).__init__()

		self.clip_model = AutoModel.from_pretrained("google/siglip2-so400m-patch14-384", torch_dtype=torch.float32)
		self.clip_model = self.clip_model.vision_model

		# Replace patch embedding with 28x28 patches
		self.clip_model.embeddings.patch_embedding = nn.Conv2d(
			in_channels=3,
			out_channels=self.clip_model.config.hidden_size,
			kernel_size=28,
			stride=28,
			padding='valid',
		)
		self.resolution = 756

		self.head = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(self.clip_model.config.hidden_size, 1, bias=True),
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		embedding = self.clip_model(x).pooler_output
		return self.head(embedding).squeeze(-1)
	
	def get_preprocessor(self):
		def processor(images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
			image = images.convert("RGB")
			if image.size != (self.resolution, self.resolution):
				image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
			
			new_image = Image.new("RGB", (self.resolution, self.resolution), (128, 128, 128)) # type: ignore
			new_image.paste(image, (0, 0))

			pixel_values = TVF.pil_to_tensor(new_image) / 255.0           # [0, 1]
			pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])  # [-1, 1]

			return {"pixel_values": pixel_values.unsqueeze(0)}
		return processor


class So400m512(nn.Module):
	def __init__(self, dropout: float, full_train: bool):
		super(So400m512, self).__init__()

		self.clip_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-512", torch_dtype=torch.float32)
		self.clip_model = self.clip_model.vision_model
		self.head = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(self.clip_model.config.hidden_size, 1, bias=True),
		)

		if not full_train:
			self.clip_model.requires_grad_(False)
			#self.clip_model.head.requires_grad_(True)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		embedding = self.clip_model(x).pooler_output
		return self.head(embedding).squeeze(-1)
	
	def get_preprocessor(self):
		def processor(images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
			image = images.convert("RGB")
			if image.size != (512, 512):
				image = image.resize((512, 512), Image.Resampling.BICUBIC)  #Image.Resampling.LANCZOS)   # I got better loss with BICUBIC here
			
			new_image = Image.new("RGB", (512, 512), (128, 128, 128)) # type: ignore
			new_image.paste(image, (0, 0))

			pixel_values = TVF.pil_to_tensor(new_image) / 255.0           # [0, 1]
			pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])  # [-1, 1]

			return {"pixel_values": pixel_values.unsqueeze(0)}
		return processor


class So400m512Linear(nn.Module):
	def __init__(self, dropout: float):
		super(So400m512Linear, self).__init__()

		self.clip_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-512", torch_dtype=torch.float32)
		self.clip_model = self.clip_model.vision_model
		self.clip_model.requires_grad_(False)
		self.head = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(self.clip_model.config.hidden_size, 1, bias=True),
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		embedding = self.clip_model(x).pooler_output
		return self.head(embedding).squeeze(-1)
	
	def get_preprocessor(self):
		def processor(images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
			image = images.convert("RGB")
			if image.size != (512, 512):
				image = image.resize((512, 512), Image.Resampling.BICUBIC)  #Image.Resampling.LANCZOS)
			
			new_image = Image.new("RGB", (512, 512), (128, 128, 128)) # type: ignore
			new_image.paste(image, (0, 0))

			pixel_values = TVF.pil_to_tensor(new_image) / 255.0           # [0, 1]
			pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])  # [-1, 1]

			return {"pixel_values": pixel_values.unsqueeze(0)}
		return processor


class So400m512MLP(nn.Module):
	def __init__(self, dropout: float, full_train: bool):
		super(So400m512MLP, self).__init__()

		assert not full_train, "So400m512MLP is for frozen backbone only"

		self.clip_model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-512", torch_dtype=torch.float32)
		self.clip_model = self.clip_model.vision_model
		self.clip_model.requires_grad_(False)
		self.head = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(self.clip_model.config.hidden_size, self.clip_model.config.hidden_size * 2, bias=True),
			nn.GELU(),
			nn.Linear(self.clip_model.config.hidden_size * 2, 1, bias=True),
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		embedding = self.clip_model(x).pooler_output
		return self.head(embedding).squeeze(-1)
	
	def get_preprocessor(self):
		def processor(images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
			image = images.convert("RGB")
			if image.size != (512, 512):
				image = image.resize((512, 512), Image.Resampling.BICUBIC)  #Image.Resampling.LANCZOS)
			
			new_image = Image.new("RGB", (512, 512), (128, 128, 128)) # type: ignore
			new_image.paste(image, (0, 0))

			pixel_values = TVF.pil_to_tensor(new_image) / 255.0           # [0, 1]
			pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])  # [-1, 1]

			return {"pixel_values": pixel_values.unsqueeze(0)}
		return processor


class SiglipB16_512(nn.Module):
	def __init__(self, dropout: float):
		super(SiglipB16_512, self).__init__()

		self.clip_model = AutoModel.from_pretrained("google/siglip2-base-patch16-512", torch_dtype=torch.float32)
		self.clip_model = self.clip_model.vision_model
		self.head = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(self.clip_model.config.hidden_size, 1, bias=True),
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		embedding = self.clip_model(x).pooler_output
		return self.head(embedding).squeeze(-1)
	
	def get_preprocessor(self):
		def processor(images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
			image = images.convert("RGB")
			if image.size != (512, 512):
				image = image.resize((512, 512), Image.Resampling.LANCZOS)
			
			new_image = Image.new("RGB", (512, 512), (128, 128, 128)) # type: ignore
			new_image.paste(image, (0, 0))

			pixel_values = TVF.pil_to_tensor(new_image) / 255.0           # [0, 1]
			pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])  # [-1, 1]

			return {"pixel_values": pixel_values.unsqueeze(0)}
		return processor


class RescaledClipL(nn.Module):
	def __init__(self, dropout: float):
		super(RescaledClipL, self).__init__()

		clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336", dtype=torch.float32)
		self.clip_model = clip_model.vision_model

		# Interpolate position embeddings to 532x532 (38x38 patches = 1444+1)
		with torch.no_grad():
			dtype = self.clip_model.embeddings.position_embedding.weight.dtype
			interpolated_embeddings = self.clip_model.embeddings.interpolate_pos_encoding(torch.zeros(1, 1445, 1024, dtype=dtype), 532, 532)
			logging.info(f"Rescaled position embeddings from {self.clip_model.embeddings.position_embedding.weight.shape} to {interpolated_embeddings.shape}")
			new_pos_emb = nn.Embedding(interpolated_embeddings.shape[1], self.clip_model.config.hidden_size).to(dtype=dtype)
			new_pos_emb.weight.copy_(interpolated_embeddings.squeeze(0))
		self.clip_model.embeddings.position_embedding = new_pos_emb
		self.clip_model.embeddings.register_buffer("position_ids", torch.arange(interpolated_embeddings.shape[1]).expand((1, -1)), persistent=False)
		self.clip_model.embeddings.image_size = 532
		self.resolution = 532

		self.head = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(self.clip_model.config.hidden_size, 1, bias=True),
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		embedding = self.clip_model(x).pooler_output
		return self.head(embedding).squeeze(-1)
	
	def get_preprocessor(self):
		def processor(images: Image.Image, return_tensors: str) -> dict[str, torch.Tensor]:
			image = images.convert("RGB")
			small_side = min(image.size)
			if small_side != self.resolution:
				scale = self.resolution / small_side
				image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
			assert min(image.size) == self.resolution
			
			new_image = Image.new("RGB", (self.resolution, self.resolution), (128, 128, 128)) # type: ignore
			excess_x = (self.resolution - image.width) // 2
			excess_y = (self.resolution - image.height) // 2
			new_image.paste(image, (excess_x, excess_y))

			pixel_values = TVF.pil_to_tensor(new_image) / 255.0           # [0, 1]
			pixel_values = TVF.normalize(pixel_values, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

			return {"pixel_values": pixel_values.unsqueeze(0)}
		return processor


class PairwiseImages(Dataset):
	def __init__(self, pairs: list[tuple[str, str, int, float]], processor, cache_images: bool, cache_dir: Path):
		self.pairs = pairs
		self.processor = processor
		self.cache_images = cache_images
		self.cache_namespace = callable_fingerprint(processor)
		self.cache_dir = Path(cache_dir) / f"proc-{self.cache_namespace}"
		if self.cache_images:
			self.cache_dir.mkdir(parents=True, exist_ok=True)
	
	def __len__(self):
		return len(self.pairs)
	
	def __getitem__(self, idx):
		path1, path2, label, dist = self.pairs[idx]

		pixel_values1 = self.load_and_process_image(path1)
		pixel_values2 = self.load_and_process_image(path2)

		# For now, sort so winner is always left
		if label == -1:
			pixel_values1, pixel_values2 = pixel_values2, pixel_values1
			label = 1
		elif label == 0:
			raise ValueError("Ties are not supported in this dataset")

		return torch.stack((pixel_values1, pixel_values2), dim=0), torch.tensor(label, dtype=torch.long), torch.tensor(dist, dtype=torch.float32)
	
	def _cache_path(self, path: Path | str) -> Path:
		path = Path(path)
		stat = path.stat()
		key = hashlib.sha256(f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode()).hexdigest()
		return self.cache_dir / f"{key}.pt"
	
	def load_and_process_image(self, path: str | Path) -> torch.Tensor:
		cpath = self._cache_path(path)
		if self.cache_images:
			if cpath.exists():
				try:
					return torch.load(cpath)
				except Exception as e:
					logging.warning(f"Failed to load cached image tensor from {cpath}: {e}")

		img = Image.open(path).convert('RGB')
		pixel_values = self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)

		if self.cache_images:
			with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False) as tf:
				torch.save(pixel_values, tf.name)
				Path(tf.name).rename(cpath)

		return pixel_values


class SingleImages(Dataset):
	def __init__(self, paths: list[str], processor):
		self.paths = paths
		self.processor = processor
	
	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, idx):
		img1 = Image.open(self.paths[idx]).convert('RGB')

		pixel_values1 = self.processor(images=img1, return_tensors="pt")['pixel_values'].squeeze(0)

		return pixel_values1



def soft_tie_loss(scores_a, scores_b, labels, tau=1.0, gamma=1.0, lambda_tie=0.1):
	"""
	Implements soft-tie calibration for pairwise ranking.

	Args:
		scores_a, scores_b: model outputs (N,)
		labels: tensor of shape (N,) with values:
			1 → A wins, 0 → B wins, -1 → tie
		tau: temperature for logistic scaling
		gamma: width of soft-tie band (controls confidence smoothing)
		lambda_tie: penalty weight for enforcing |Δs| ≈ 0 on tie labels
	"""
	delta = (scores_a - scores_b) / tau
	base_prob = torch.sigmoid(delta)
	alpha = torch.exp(-torch.abs(delta) / gamma)  # stronger smoothing near small margins
	soft_prob = alpha * 0.5 + (1 - alpha) * base_prob

	# CE for non-ties
	mask_non_tie = (labels != -1)
	y_true = (labels == 1).float()
	loss_ce = F.binary_cross_entropy(soft_prob[mask_non_tie], y_true[mask_non_tie])

	# Band penalty for tie pairs
	mask_tie = (labels == -1)
	loss_tie = torch.mean(F.huber_loss(torch.abs(delta[mask_tie]), torch.zeros_like(delta[mask_tie])))

	return loss_ce + lambda_tie * loss_tie




######### Image Corruption
def jpeg_compression_augment(img: Image.Image, rng) -> tuple[Image.Image, Image.Image] | None:
	# First, compress at 90% to get a baseline
	buf = io.BytesIO()
	img.save(buf, format="JPEG", quality=90)
	baseline = len(buf.getvalue())
	# Then walk the quality down until we get to 33% or less data
	for quality in range(90, 10, -5):
		buf = io.BytesIO()
		img.save(buf, format="JPEG", quality=quality)
		if len(buf.getvalue()) <= baseline * 0.33:
			compressed_img = Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
			return img, compressed_img
	
	return None


def lowres_augment(img: Image.Image, rng) -> tuple[Image.Image, Image.Image] | None:
	# The maximum resolution we care about for IQA is 1024px, so to ensure a fair comparison,
	# we first downscale the "original" image to max 1024px side if needed.
	max_side = max(img.size)
	if max_side > 1024:
		scale = 1024 / max_side
		new_size = (int(img.width * scale), int(img.height * scale))
		img = img.resize(new_size, Image.LANCZOS)
	
	scale_down = rng.uniform(0.5, 0.9)
	lowres_size = (max(1, int(img.width * scale_down)), max(1, int(img.height * scale_down)))
	lowres_img = img.resize(lowres_size, Image.LANCZOS)
	upscale_method = rng.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
	upscaled_img = lowres_img.resize(img.size, upscale_method)

	return img, upscaled_img


class CorruptPairwiseImages(Dataset):
	def __init__(self, paths: list[str], processor, corrupter):
		self.paths = paths
		self.processor = processor
		self.corrupter = corrupter
	
	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, idx):
		img = Image.open(self.paths[idx]).convert('RGB')
		rng = random.Random(idx)
		result = self.corrupter(img, rng)
		if result is None:
			pixel_values = self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
			return torch.stack((pixel_values, pixel_values), dim=0), torch.tensor(1, dtype=torch.long), False
		
		clean_img, corrupt_img = result

		pixel_values1 = self.processor(images=clean_img, return_tensors="pt")['pixel_values'].squeeze(0)
		pixel_values2 = self.processor(images=corrupt_img, return_tensors="pt")['pixel_values'].squeeze(0)

		return torch.stack((pixel_values1, pixel_values2), dim=0), torch.tensor(1, dtype=torch.long), True



def _get_cosine_schedule_with_warmup_lr_lambda(
	current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_ratio: float
):
	if current_step < num_warmup_steps:
		return float(current_step) / float(max(1, num_warmup_steps))
	
	progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
	r = 1.0 - min_lr_ratio
	return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * r + min_lr_ratio


def get_cosine_schedule_with_warmup(
	optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, min_lr_ratio: float = 0.0
):
	lr_lambda = partial(
		_get_cosine_schedule_with_warmup_lr_lambda,
		num_warmup_steps=num_warmup_steps,
		num_training_steps=num_training_steps,
		num_cycles=num_cycles,
		min_lr_ratio=min_lr_ratio,
	)
	return LambdaLR(optimizer, lr_lambda, last_epoch)




####### Processor Fingerprinting
def _safe_repr(x):
	try:
		return repr(x)
	except Exception:
		return f"<unrepr:{type(x).__name__}>"

def _module_version(mod_name):
	try:
		return md.version(mod_name.split('.',1)[0])
	except Exception:
		return None

def _class_path(obj):
	cls = obj.__class__
	return f"{cls.__module__}.{cls.__name__}"

def _serialize_function(fn):
	data = {
		"type": "function",
		"module": getattr(fn, "__module__", None),
		"qualname": getattr(fn, "__qualname__", None),
		"name": getattr(fn, "__name__", None),
		"filename": None,
		"firstlineno": None,
		"signature": None,
		"defaults": None,
		"kwdefaults": None,
		"closure": None,
		"bytecode": None,
	}
	try:
		data["signature"] = str(inspect.signature(fn))
	except Exception:
		pass
	if hasattr(fn, "__code__"):
		co = fn.__code__
		data["filename"] = co.co_filename
		data["firstlineno"] = co.co_firstlineno
		# Bytecode is not guaranteed stable across Python versions — we also add versions below.
		data["bytecode"] = co.co_code.hex()
	if getattr(fn, "__defaults__", None) is not None:
		data["defaults"] = [_safe_repr(x) for x in fn.__defaults__]
	if getattr(fn, "__kwdefaults__", None) is not None:
		data["kwdefaults"] = {k: _safe_repr(v) for k, v in fn.__kwdefaults__.items()}
	if getattr(fn, "__closure__", None) is not None:
		data["closure"] = [_safe_repr(c.cell_contents) for c in fn.__closure__ if c is not None]
	return data

def _serialize_method(m):
	# Bound methods: include function + owning class path
	return {
		"type": "method",
		"owner": _class_path(m.__self__) if hasattr(m, "__self__") else None,
		"function": _serialize_function(m.__func__)
	}

def _serialize_partial(p):
	return {
		"type": "partial",
		"func": serialize_callable(p.func),
		"args": [_safe_repr(a) for a in p.args],
		"keywords": {k: _safe_repr(v) for k, v in (p.keywords or {}).items()},
	}

def _serialize_callable_instance(obj):
	# Callable object: capture class and a sanitized view of __dict__
	d = {}
	for k, v in getattr(obj, "__dict__", {}).items():
		if callable(v) or isinstance(v, types.ModuleType):
			continue
		try:
			json.dumps(v)
			d[k] = v
		except TypeError:
			d[k] = _safe_repr(v)
	return {
		"type": "callable_instance",
		"class": _class_path(obj),
		"state": d,
	}

def serialize_callable(proc):
	# functools.partial
	if isinstance(proc, functools.partial):
		return _serialize_partial(proc)

	# functions / builtins / methods
	if isinstance(proc, types.FunctionType):
		return _serialize_function(proc)
	if isinstance(proc, types.MethodType):
		return _serialize_method(proc)
	if isinstance(proc, (types.BuiltinFunctionType, types.BuiltinMethodType)):
		return {
			"type": "builtin",
			"module": getattr(proc, "__module__", None),
			"name": getattr(proc, "__name__", None),
		}

	# Any callable instance (has __call__)
	if callable(proc):
		# If __call__ is a function or method, include that too
		call_attr = getattr(proc, "__call__", None)
		call_ser = _serialize_method(call_attr) if isinstance(call_attr, types.MethodType) else None
		base = _serialize_callable_instance(proc)
		if call_ser:
			base["call"] = call_ser
		return base

	# Fallback: just repr
	return {"type": "unknown", "repr": _safe_repr(proc)}

def callable_fingerprint(proc, extra_salts=None):
	"""Return a stable hex digest that changes when the processor’s behavior is likely to change."""
	payload = {
		"callable": serialize_callable(proc),
		"env": {
			"python": sys.version.split()[0],
			"packages": {
				# add more here if they can affect preprocessing
				"Pillow": _module_version("PIL") or _module_version("Pillow"),
				"torch": _module_version("torch"),
				"torchvision": _module_version("torchvision"),
				"transformers": _module_version("transformers"),
			},
		},
	}
	if extra_salts:
		payload["extra"] = extra_salts
	blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
	return hashlib.sha256(blob).hexdigest()




@hydra.main(version_base=None, config_path=None, config_name="trainer_config")
def main(config: TrainerConfig) -> None:
	"""Hydra wrapper for train."""
	if not config:
		raise ValueError("Config is empty. Please provide a valid config.")
	trainer = Trainer(config)
	return trainer.fit()


if __name__ == '__main__':
	main()
