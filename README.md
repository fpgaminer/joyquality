# JoyQuality

JoyQuality is an open source Image Quality Assessment (IQA) model.  It takes as input an image and gives as output a scalar score representing the overall quality of the image.  Highlights:

* Use for image dataset filtering.
* Use for image quality tagging, such as for training text-to-image models.
* Input size of 512x512.
* 400M parameters.
* Fast (200 images/s on a 300W H100).
* A diverse and balanced dataset to ensure robust and adaptability.
* Quickly learns new, tiny preference datasets.


## What's Cool About JoyQuality

Let's say you have a dataset of images and you want to assign each an overall quality score.  Normally trying to score/rank images can be _extremely_ difficult.  Even if your dataset is small (1000 images), performing ELO ranking can take a lot of manual comparisons (10k in the case of 1k images).  As someone who has done a lot of manual image vs image comparisons, each comparison is no easy feat.  There are many factors to consider: framing, execution quality, resolution, compression artifacts, etc.  All while trying to be unbiased with respect to content and subject.

Instead, you could train a model to do the scoring based on a small pairwise preference dataset.  Then use that model to score your overall dataset.  But I know from experience that even at 5k pairwise preferences, you can only get _decent_ results from finetuning an existing vision model (like CLIP).

Additionally, training these types of models is surprisingly subtle.  Everything from: the selection of images to compare, which pairs of images to compare, building a non-poisoned validation set, metrics to measure model performance, etc.

That's where JoyQuality steps in.  JoyQuality was trained on a broad, diverse array of images and 100k carefully selected pairwise preferences, while also being tuned across a range of metrics to ensure robust performance.  By using JoyQuality as your base model, you can quickly build a small personal preference dataset (e.g. 1k pairs) and then train on top to build a robust IQA model.

As an example, on my own personal preference dataset (which was completely excluded from JoyQuality's training), JoyQuality _starts_ at 75% accuracy and finishes at 80% accuracy in only 77 steps (256 batch size).  All while also maintaining robust AURC, Brier, ECE metrics, and >98% accuracy on simple JPEG and low-resolution tasks.  (Performance on the latter two tasks is easy to lose on poorly trained models).  When trained from so400m-512 directly instead of JoyQuality, various robustness metrics such as the JPEG and low resolution tasks only got to 90% and 80% respectively.


## The Model

JoyQuality is built on top of SigLIP 2's so400m-512 model.  This is an exceptionally well trained vision model from Google, similar to OpenAI's CLIP, but with much better finetuning dynamics.  Unlike older Aesthetic models like the ones used for SD1.5 and SDXL, JoyQuality has an input resolution of 512x512, enabling it to see and differnetiate small details.

During training, both JPEG and low-resolution validation tasks were measured.  These tasks were built by taking the set of validation images and corrupting them, building simple pairs of original-vs-corrupt pairs to test the model against.  For JPEG corruption, the image is compressed until it loses at least 70% of its information.  For resolution corruption, images are downscaled to between 50% and 90% of its original size, before being scaled back up using a randomly selected upscaling algorithm.  This simple benchmark is able to quickly differentiate models with a low input resolution (such as CLIP's 224 or 336), which struggle especially with the low resolution test.  JoyQuality, however, achieves over 99% accuracy.

Using the so400m architecture also gives JoyQuality plenty of general performance compared to smaller models like CLIP-L or CLIP-B.  so400m outperformed both SigLIP-B and SigLIP-L in my ablation tests, with so400m achieving 80% accuracy compared to SigLIP-B's 70%, while being faster than SigLIP-L.

Despite being 400M parameters and high resolution (512x512), JoyQuality's processing speed is _fast_.  On a 300W H100 I was able to run the model in inference mode at 204 images/s.  Just 19 hours to process through the 14M images of bigASP 2.5's dataset.

Finally, the so400m architecture doesn't crop input images, unlike CLIP, which means JoyQuality sees the _entire_ image regardless of aspect ratio.


## Usage

I highly recommend tuning the model on your own set of preference pairs.  JoyQuality is designed to be finetuned and, importantly, everyone grades things differently.  All you need is image-vs-image comparisons with one labelled as a "winner".  I'd recommend 2k minimum for great results, though you can certainly get decent results with less.

### Finetuning

**WARNING** Usage is a PITA right now. The training script is total garbage until I can clean it up and make it friendlier.

For the current training script you need your dataset in this format:

```
{
  "train": [
    [
      "/example/path/train/imageA1.jpg",
      "/example/path/train/imageA2.jpg",
      1
    ],
    [
      "/example/path/train/imageB1.jpg",
      "/example/path/train/imageB2.jpg",
      1
    ],
    [
      "/example/path/train/imageC1.jpg",
      "/example/path/train/imageC2.jpg",
      1
    ]
  ],
  "test": [
    [
      "/example/path/test/imageX1.jpg",
      "/example/path/test/imageX2.jpg",
      1
    ],
    [
      "/example/path/test/imageY1.jpg",
      "/example/path/test/imageY2.jpg",
      1
    ]
  ]
}
```

Basically two paths for each entry plus a label.  **For now the training script only supports the winning image being the first image, and the losing image being the second, and it ignores the label.**  Your images don't have to be a particular size/format/etc, as long as PIL can load them.  The general rule of thumb is to have 10% test data.

Then you can run the training script like this:

`python train.py lr_scheduler=onecycle learning_rate=0.008 warmup_samples=1500 batch_size=256 total_samples=20000 model=So400m512 base_checkpoint=o8eg1n4c.safetensors dataset=your-dataset.json wandb_project=... device_batch_size=4`

Those are the settings I found to work best for my personal preference dataset which has about 6k pairs in it.  The model both learns quickly and overfits quickly, so if your dataset is smaller definitely decrease `total_samples`.  You'll want to adjust `device_batch_size` based on how much GPU memory you have (it doesn't have any effect on the real batch size; the script automatically accumulates to reach the target).

`cosine` schedule is the "standard", but at least in my case `onecycle` edged out ahead.  One Cycle tends to work quite well for low data, high repeat regimes.  Though expect to adjust the learning rate if you switch the schedule.

### Inference

The [score-images.py](./score-images.py) script gives an example of how to use the model during inference.  In my case I use a SQL database to keep track of everything, so the script is set up that way.  But your setup will likely be different.  Adjust as needed.

The model outputs a "latent score" for each image.  I'll just call it the image's score.  It's an unscaled, unbiased number, which means it doesn't really "mean" anything out of context.  You can compare two images by doing:

`sigmoid(image_a_score - image_b_score)`

That results in a value between 0.0 and 1.0 which represent the probability that Image A would beat Image B in a contest.

If you have a dataset of images you can also use the scores to just directly rank them all from best to worst.  Just sort by their scores.  The best image will have the highest score; the worst the lowest.

Many text-to-image diffusion training processes want images graded into discrete levels like "worst quality", "low quality", "average quality", "high quality", "best quality", etc.  You can use the score to do this as well!  You just have to break up the range of scores in your dataset into distinct buckets from best to worst.  First score all your images.  Then use one of two approaches to convert the scores into discrete "ranks."

#### Ranking Method A

This is the simple approach.  Get the min and max score in your dataset and divide the range evenly by however many buckets you need.  Done!  If you're training a text-to-image model this simple method helps ensure that each quality is equally represented, which can help the model learn those signals more robustly.

The downside is that if your dataset is highly imbalanced, this method will be imperfect.  i.e. the upper 10% of images in your dataset might not all be high quality.  Yet if you divide the range evenly by 10 buckets then all those images will be labelled high quality.

#### Ranking Method B

This method is a little more complicated, but the goal is to help preserve the relative probabilities that the model natively spits out.  As you may recall from above, you can compare two images using the model's score, and get a probability that Image A beats Image B.  i.e. it can tell you not only that Image A is better, but by _how much_.  Method B helps preserve this information and thus better handles highly imbalanced datasets.  That way each bucket will genuinely contain images of related overall quality.  No "very high quality" bleeding into "best quality".

So, method B uses this equation to calculate the rank of an image from its score:

`rank = 10 * (1 / (1 + torch.exp(-(s - b) / tau)))`

Where `s` is the score, and `b` and `tau` are parameters that get tuned to your specific dataset.  They're basically "scale" and "offset" variables.  You'll need a small validation set of preferences to tune those parameters.  If you finetuned JoyQuality you can use the test set from there.  Otherwise just use a random subset of your dataset.  The optimization is the same as during finetuning, pushing the model to match your dataset's preferences, except now it's working to optimize tau.  After that `b` is found using a simple search.  All of this basically helps to calibrate the scores post-training and fit them into a global range of (0, 1).

After tuning `tau` and `b` you can use the formula above, which will spit out a continuous rank between 0 and 10 for each image.  You can divide that up however you'd like.  In my case I have ten quality buckets, so I just do `int(rank)` and get ranks [0, 9] inclusive, which I later convert to text labels.

Sound complicated?  Here's the code:

```
def fit_tau_torch(s: torch.Tensor, pairs: torch.Tensor, y: torch.Tensor, tau0: float = 1.0, steps: int = 200, lr: float = 0.1) -> float:
	"""
	Minimizes BCE(y, sigmoid((s_i - s_j) / tau)) over pairs to estimate tau.
	"""
	assert pairs.dtype == torch.long
	assert pairs.ndim == 2 and pairs.shape[1] == 2
	assert pairs.min() >= 0 and pairs.max() < s.shape[0]

	# Parameterize tau > 0 as tau = exp(log_tau)
	log_tau = torch.tensor([math.log(max(tau0, 1e-6))], dtype=torch.float32, requires_grad=True)
	opt = torch.optim.Adam([log_tau], lr=lr)
	
	M = pairs.shape[0]

	for t in range(steps):
		si = s[pairs[:, 0]]
		sj = s[pairs[:, 1]]
		logits = (si - sj) / log_tau.exp()

		loss = F.binary_cross_entropy_with_logits(logits, y.float(), reduction='mean')
		opt.zero_grad(set_to_none=True)
		loss.backward()
		opt.step()

		print(f"[{t+1:4d}/{steps}] loss={loss.item():.6f} tau={log_tau.exp().item():.6f}")

	tau = float(log_tau.exp().item())
	return tau


def fit_tau_torch_lbfgs(s: torch.Tensor, pairs: torch.Tensor, y: torch.Tensor, tau0: float = 1.0, max_iter: int = 50) -> float:
	# log_tau parameterization keeps tau > 0
	log_tau = torch.tensor([math.log(max(tau0, 1e-6))], dtype=torch.float32, requires_grad=True)
	opt = torch.optim.LBFGS([log_tau], max_iter=max_iter, line_search_fn="strong_wolfe")

	si = s[pairs[:, 0]]
	sj = s[pairs[:, 1]]

	last_loss = None
	def closure():
		nonlocal last_loss
		opt.zero_grad(set_to_none=True)
		tau = log_tau.exp()
		logits = (si - sj) / tau
		loss = F.binary_cross_entropy_with_logits(logits, y.float(), reduction='mean')
		loss.backward()
		last_loss = loss.detach()
		print(f"loss={loss.item():.6f} tau={tau.item():.6f}")
		return loss

	opt.step(closure)

	return float(log_tau.exp().item())


def solve_b_for_mean(s, tau, target_mean: float, iters: int = 50):
	# find b such that mean(9 * sigmoid((s - b) / tau)) = target_mean
	lo, hi = torch.min(s) - 10 * tau, torch.max(s) + 10 * tau

	for _ in range(iters):
		b = 0.5 * (lo + hi)
		scores = 9 * (1 / (1 + torch.exp(-(s - b) / tau)))
		if scores.mean() > target_mean:
			lo = b
		else:
			hi = b
		print(f"b in [{lo.item():.6f}, {hi.item():.6f}], mean score = {scores.mean().item():.6f}")

		if hi - lo < 1e-6:
			break
	
	return float(0.5 * (lo + hi))


human_pairs = json.loads(Path("pairs-dataset-human.json").read_text())
validation_pairs = human_pairs['test']
validation_pairs = [(path_to_index[a], path_to_index[b]) for a, b, y in human_pairs['test']]
validation_labels = [y for a, b, y in human_pairs['test']]

print(f"# Validation pairs: {len(validation_pairs)}")

# Prepare tensors
s = torch.tensor([filehash_to_bt_score[fh] for fh in all_filehashes], dtype=torch.float32)
pairs = torch.tensor(validation_pairs, dtype=torch.long)
y = torch.tensor(validation_labels, dtype=torch.float32)

# Solve for tau
#tau = fit_tau_torch(s, pairs, y, lr=0.05)
tau = fit_tau_torch_lbfgs(s, pairs, y)
print(f"Fitted tau: {tau}")

# Solve for b
b = solve_b_for_mean(s, tau, target_mean=5.0)
print(f"Solved b: {b}")

# Compute ranks
ranks = 10 * (1 / (1 + torch.exp(-(s - b) / tau)))

print(f"Ranks stats: min={ranks.min().item():.6f}, max={ranks.max().item():.6f}, mean={ranks.mean().item():.6f}, std={ranks.std().item():.6f}")
```


## What is Image Quality Assessment

Image Quality Assessment (IQA) is a whole field, and there are many pre-existing models built on public IQA datasets.  If you want to know more about IQA, it's a field well worth looking into.  Which all raises the question ... why build JoyQuality if IQA models already exist?  For me, personally, I found the existing IQA models to be somewhat obtuse to use.  They also only seemed to come in odd sizes; tiny or huge.  There are IQA models built on top of VLMs which have excellent performance on the public IQA datasets while also being able to break down different aspects of the image's quality.  Those kinds of models are the "future", but are for now too heavyweight in my use cases.  So I just built my own :P


## How It Was Built

### The Dataset
JoyQuality's approach is to act as a base model.  It does not itself need to be the absolute "best" model in terms of accuracy, since end users are expected to do a quick finetune on their own, much smaller, dataset.  But it does need to be a robust base, so that it can quickly adapt to new data.

So this defines our parameters for JoyQuality's dataset.  To make the model adaptable and robust, the dataset needs to be diverse, balanced, and well constructed.  But the accuracy of the pairwise comparisons themselves is less important.

I started out by taking my large pool of 14M images, and run them all through CLIP-B to get embeddings.  From these embeddings I could then build a smaller dataset for JoyQuality that was balanced by maximizing the CLIP distance between the images in the dataset.  i.e. seed the dataset with a single, randomly selected image.  Then iterate over and over again, picking another image from the pool at each step by finding the image in the pool that is furthest from all images currently in the dataset.

From this set of images I then carefully built pairings.  There are a few approaches to picking pairings: completely randomly; hub and spoke; fixed k.  (Hub and spoke is where you have a small subset of the images that appear very frequently, with those images forming hubs that all the other images connect to through pairings).  A good amount of research has shown fixed k to perform best in practice.  This approach just sets how many times each image will appear in pairings.  In my case I picked 4.  So from a dataset of 25k images, I ended up with 100k pairings.

But there's one more twist.  Beyond ensuring that each image is represented 4 times, we still need to pick their partners.  Again that could be completely random, but instead I did a split approach.  30% of the time it was random.  70% of the time an image was picked based on minimizing CLIP distance.  What this accomplishes is that 70% of the pairings are between images that are similar to each other.  Why do that?  The theory is that by training a model on pairs of images that are similar in CLIP space, the model will be forced to learn that even though the content of the images is the same, the quality might be different, and to amplify that quality signal.

NOTE: I'm not sure if this is an _ideal_ approach yet, but the theory makes some sense.  The concern of course is that CLIP space definitely already has some quality features in it, so by minimizing CLIP distance we're making the pairs hard not only in terms of content, but also quality.  In hindsight I'd probably lower the ratio to at least 50/50, or mix in other approaches of picking pairings.  (e.g. you can train an early quality model, and then use an algorithm like OpenSkill to pick the "best" pairings and expand the dataset).

With a solid, well built set of images and pairings in hand we have accomplished the most crucial part of JoyQuality's dataset.  Now for the actual preference data!  As established earlier, it wasn't critical that these be of the highest possible accuracy.  The approach is rather _quantity_ over quality, since quantity helps make the model more robust and adaptable.

Using a set of preferences I built by hand I was able to run a large scale experiment I called the "Model Agreement" experiment.  This is where I run that small set of pairings through all of the public SOTA vision capable LLMs, using a well crafted prompt.  I could then measure what percentage of the time those models agreed with my own preferences.  In other words, this allowed me to assess systematically which of these LLMs was the strongest at this particular task.  I tested various models from Qwen, OpenAI, Google, and Anthropic.  GPT-5 mini (minimal or low thinking) was by far the best (better than the normal GPT-5).  Which is nice, because it is extremely cheap.  About $0.002 USD per call for this use.  Meaning just $200 to build a dataset of 100k preferences.

In addition to finding the best performing LLM, I also spent a good amount of time refining the prompt used.  To assist in this process I did an AI assisted prompt optimization loop.  This is where I ran the current prompt through Model Agreement to measure its performance, and then had a big LLM edit the prompt to try and improve the performance.  At each step of the loop the big LLM is given all previous prompts along with their performances, thus enabling it to "optimize" the prompt by experimenting and seeing what affects performance.  (N.B. At each loop I changed the set of preferences used for Model Agreement to prevent reward hacking).  Importantly every prompt includes instructions to output not only the preference, but the _reason_ for the preference.  I could then include some of these "reasoning traces" along with the prompt's performance, enabling the big LLM to hopefully have deeper insight into exactly _where_ and _why_ the current prompt is failing.

This prompt optimization process is very helpful when your runtime model is small (which, in this case, GPT-5 mini is).  The bigger models like GPT-5 Thinking and Claude Sonnet 4.5 are much, much better at infering things about your instructions.  So using them to expand upon a prompt and make it more detailed helps a lot in getting performance out of the smaller models.

The optimized prompt used to build JoyQuality's dataset is available here: [DATASET_PROMPT13.py](./DATASET_PROMPT13.py)

Much compute later and the dataset was ready.

### Training

I spent a long time A-B testing various models to finetune JoyQuality from: various CLIPs, SigLIPs, OWLv2, and DINOv3.  For each one I spent time optimzing hyperparameters so they all performed their best.  All in all, so400m came out on top.  DINOv3 has a larger base resolution (>1024x1024) but just didn't perform as well as even CLIP (with a much smaller resolution).  The same goes for OWLv2, which I suspect is because it was heavily tuned for object recognition, killing off its quality signals.

In addition to picking the strongest pretrained model I also evaluated different architecture tweaks.  All of these models output an embedding, so in all cases at least a new head had to be inserted to project from the embedding to a score scalar.  But there are many ways to do this: add a new layer at the end, replace the existing head, etc.  And when adding a new head, the head could be a simple linear projection, an MLP, a transformer layer, etc.  The best performing was simply a linear projection.  This makes sense, but it was worth double checking.

I also did some experiments on CLIP where I would replace/retrain its input layer to handle larger resolutions.  None of these panned out after some work.  (It's certainly possible, but I think it would require more data and more work).

Once the base model was picked (SigLIP 2 so400m-512/16), I spent even more time tweaking hyperparameters for the larger run.  In the end:

* AdamW
* Learning Rate: 0.00002
* Weight Decay: 0.01
* Batch Size: 256
* Cosine schedule
* 100,000 training samples seen
* 4,000 warmup samples

### Results

I added many, many metrics to monitor the "overall" performance of the model.  Accuracy alone is a rather shallow metric, so I wanted to keep my eye on other things.  You can view all of the graphed results here: https://api.wandb.ai/links/hungerstrike/bdka9qid

* The "weighted" variations are weighted based on CLIP distance between the pairs, so they should be more "balanced" and representative of performance in practice.  (This was needed because the dataset skews towards close pairs, whereas real world datasets won't.)
* lowres_accuracy tests the model's ability to score low resolution images lower.  Similar for jpeg_accuracy.
* Spearman Rho tests the model's ability to actually rank images.  I took a small part of the validation dataset and ran more pairwise comparisons on them until they could be perfectly ranked from best to worst.  The model does the same, and those ranked lists are compared.  This is a useful metrics for saying "Okay, the model only has X accuracy on head-to-head comparisons, but how well does it do at overall ranking, even if individual pairs aren't perfect?"
* Kendall Tau does a similar ranking-vs-ranking test, but pays more attention to the local structure of the ranking.
* ECE is an odd metric, but generally should be low.  If it grows it means the model is becoming overconfident.
* The rest of the metrics ... I dunno good interpretations of them honestly.  I found the dec_* metrics to be useless.  Brier is kind of a better accuracy measure, but I didn't look too closely at it.
* Overall I just liked to see NLL going down, ECE low, Accuracy high, and Spearman high.
