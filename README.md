# JoyQuality

JoyQuality is an open source Image Quality Assessment (IQA) model.  It takes as input an image and gives as output a scalar score.  This score can be used to either globally rank an image (how good is this image compared to all images) or to pairwise compare it to another image.

This model can be used for:

* Dataset filtering
* Image quality tagging for training Text-to-Image models (e.g. labelling images as low quality, high quality, etc)
* Probably more, I dunno.


## What's Cool About JoyQuality

Let's say you have a dataset of images and you want to assign each an overall quality score.  Normally trying to score/rank images can be _extremely_ difficult.  Even if your dataset is small (1000 images), performing ELO ranking can take a lot of manual comparisons (10k in the case of 1k images).  As someone who has done a lot of manual image vs image comparisons, each comparison is no easy feat.  There are many factors to consider: framing, execution quality, resolution, compression artifacts, etc.  All while trying to be unbiased with respect to content and subject.

Instead, you could train a model to do the scoring based on a small pairwise preference dataset.  Then use that model to score your overall dataset.  But I know from experience that even at 5k pairwise preferences, you can only get _decent_ results from finetuning an existing vision model (like CLIP).

Additionally, training these types of models is surprisingly subtle.  Everything from: the selection of images to compare, which pairs of images to compare, building a non-poisoned validation set, metrics to measure model performance, etc.

That's where JoyQuality steps in.  JoyQuality was trained on a broad, diverse array of images and 100k carefully selected pairwise preferences, while also being carefully tuned across a range of metrics to ensure robust performance.  By using JoyQuality as your base model, you can quickly build a small personal preference dataset (e.g. 1k pairs) and then train on top to build a robust IQA model.

As an example, on my own personal preference dataset (which was completely excluded from JoyQuality's training), JoyQuality _starts_ at 75% accuracy and finishes at 80% accuracy in only 77 steps (256 batch size).  All while also maintaining robust AURC, Brier, ECE metrics, and >98% accuracy on simple JPEG and low-resolution tasks.  (Performance on the latter two tasks is easy to lose on poorly trained models).


## The Model

JoyQuality is built on top of SigLIP2's so400m-512 model.  This is an exceptionally well trained model from Google, similar to OpenAI's CLIP, but with much better finetuning dynamics.  Unlike older Aesthetic models like the ones used for SD1.5 and SDXL, JoyQuality has an input resolution of 512x512, enabling it to see and differnetiate small details.

During training, both JPEG and low-resolution validation tasks were measured.  These tasks were built by taking the set of validation images and corrupting them, building simple pairs of original-vs-corrupt pairs to test the model against.  For JPEG corruption, the image is compressed until it loses at least 70% of its information.  For resolution corruption, images are downscaled to between 50% and 90% of its original size, before being scaled back up using a randomly selected upscaling algorithm.  This simple benchmark is able to quickly differentiate models with a low input resolution (such as CLIP's 224 or 336), which struggle especially with the low resolution test.  JoyQuality, however, achieves over 99% accuracy.

Using the so400m architecture also gives JoyQuality plenty of general performance compared to smaller models like CLIP-L or CLIP-B.  so400m outperformed both SigLIP-B and SigLIP-L in my ablation tests, with so400m achieving 80% accuracy compared to SigLIP-B's 70%.

Despite being 400M parameters and high resolution (512x512), JoyQuality's processing speed is _fast_.  On a 300W H100 I was able to run the model in inference mode at 204 images/s.  Just 24 hours to process through the 14M images of bigASP 2.5's dataset.

Finally, the so400m architecture doesn't crop input images, unlike CLIP, which means JoyQuality sees the _entire_ image regardless of aspect ratio.


## Usage


## What is Image Quality Assessment

Image Quality Assessment (IQA) is a whole field, and there are many pre-existing models built on public IQA datasets.  If you want to know more about IQA, it's a field well worth looking into.  Which all raises the question ... why build JoyQuality if IQA models already exist?  For me, personally, I found the existing IQA models to be somewhat obtuse to use.  They also only seemed to come in odd sizes; tiny or huge.  There are IQA models built on top of VLMs which have excellent performance on the public IQA datasets while also being able to break down different aspects of the image's quality.  Those kinds of models are the "future", but are for now too heavyweight in my use cases.  So I just built my own :P


## How It Was Built

### The Dataset
JoyQuality's approach is to act as a base model.  It does not itself need to be the absolute "best" model in terms of accuracy, since end users are expected to do a quick fine tune on their own, much smaller, dataset.  But it does need to be a robust base, so that it can quickly adapt to new data.

So this defines our parameters for JoyQuality's dataset.  To make the model adaptable and robust, the dataset needs to be diverse, balanced, and well constructed.  But the accuracy of the pairwise comparisons themselves is less important.

I started out by taking my large pool of 14M images, and run them all through CLIP-B to get embeddings.  From these embeddings I could then build a smaller dataset for JoyQuality that was balanced by maximizing the CLIP distance between the images in the dataset.  i.e. seed the dataset with a single, randomly selected image.  Then iterate over and over again, picking another image from the pool at each step by finding the image in the pool that is furthest from all images currently in the dataset.

From this set of images I then carefully built pairings.  There are a few approaches to picking pairings: completely randomly; hub and spoke; fixed k.  (Hub and spoke is where you have a small subset of the images that appear very frequently, with those images forming hubs that all the other images connect to through pairings).  A good amount of research has shown fixed k to perform best in practice.  This approach just sets how many times each image will appear in pairings.  In my case I picked 4.  So from a dataset of 25k images, I ended up with 100k pairings.

But there's one more twist.  Beyond ensuring that each image is represented 4 times, we still need to pick their partners.  Again that could be completely random, but instead I did a split approach.  30% of the time it was random.  70% of the time an image was picked based on minimizing CLIP distance.  What this accomplishes is that 70% of the pairings are between images that are similar to each other.  Why do that?  The theory is that by training a model on pairs of images that are similar in CLIP space, the model will be forced to learn that even though the content of the images is the same, the quality might be different, and to amplify that quality signal.

NOTE: I'm not sure if this is an _ideal_ approach yet, but the theory makes some sense.  The concern of course is that CLIP space definitely already has some quality features in it, so by minimizing CLIP distance we're making the pairs hard not only in terms of content, but also quality.  In hindsight I'd probably lower the ratio to at least 50/50, or mix in other approaches of picking pairings.  (e.g. you can train an early quality model, and then use an algorithm like OpenSkill to pick the "best" pairings and expand the dataset).

With a solid, well built set of images and pairings in hand we have accomplished the most crucial part of JoyQuality's dataset.  Now for the actual preference data!  As established earlier, it wasn't critical that these be of the highest possible accuracy.  The approach is rather _quantity_ over quality, since quantity helps make the model more robust and adaptable.

Using a set of preferences I built by hand I was able to run a large scale experiment I called the "Model Agreement" experiment.  This is where I run that small set of pairings through all of the public SOTA vision capable LLMs, using a well crafted prompt.  I could then measure what percentage of the time those models agreed with my own preferences.  In other words, this allowed me to assess systematically which of these LLMs was the strongest at this particular task.  I tested various models from Qwen, OpenAI, Google, and Anthropic.  GPT-5 mini (minimal or low thinking) was by far the best (better than the normal GPT-5).  Which is nice, because it is extremely cheap.  About $0.002 USD per call for this use.  Meaning just $200 to build a dataset of 100k prefences.

In addition to finding the best performing LLM, I also spent a good amount of time refining the prompt used.  To assist in this process I did an AI assisted prompt refinement loop.  This is where I ran the current prompt through Model Agreement to measure its performance, and then had an big LLM edit the prompt to try and improve the performance.  At each step of the loop the big LLM is given all previous prompts along with their performances, thus enabling it to "optimize" the prompt by experimenting and seeing what affects performance.  (N.B. At each loop I changed the set of preferences used for Model Agreement to prevent reward hacking).  Importantly every prompt includes instructions to output not only the preference, but the _reason_ for the preference.  I could then include from of these "reasoning traces" along with the prompt's performance, enabling the big LLM to hopefully have deeper insight into exactly _where_ and _why_ the current prompt is failing.

This prompt optimization process is very helpful when your runtime model is small (which, in this case, GPT-5 mini is).  The bigger models like GPT-5 Thinking and Claude Sonnet 4.5 are much, much better at infering things about your instructions.  So using them to expand upon a prompt and make it more detailed helps a lot in getting performance out of the smaller models.

The optimized prompt used to build JoyQuality's dataset is available here: _______TODO______

Much compute later and the dataset was ready.

### Training

I spent a long time A-B testing various models to finetune JoyQuality from: various CLIPs, SigLIPs, OWLv2, and DINOv3.  For each one I spent time optimzing hyperparameters so they all performed their best.  All in all, so400m came out on top.  DINOv3 has a larger base resolution (>1024x1024) but just didn't perform as well as even CLIP (with a much smaller resolution).  The same goes for OWLv2, which I suspect is because it was heavily tuned for object recognition, so its quality signals likely died off.

In addition to picking the strongest pretrained model I also evaluated different architecture tweaks.  All of these models output an embedding, so in all cases at least a new head had to be inserted to project from the embedding to a scalar.  But there are many ways to do this: add a new layer at the end, replace the existing head, etc.  And when adding a new head, the head could be a simple linear projection, an MLP, a transformer layer, etc.  The best performing was simply a linear projection.  This makes sense, but it was worth double checking.

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

I added many, many metrics to try and monitor the "overall" performance of the model.  Accuracy alone is a rather shallow metric, so I wanted to keep my eye on other things.  You can view all of the graphed results here: https://api.wandb.ai/links/hungerstrike/bdka9qid

The "weighted" variations are weighted based on CLIP distance between the pairs, so they should be more "balanced" and representative of performance in practice.  (This was needed because the dataset skews towards close pairs, whereas real world datasets won't.)
lowres_accuracy tests the model's ability to score low resolution images lower.  Similar for jpeg_accuracy.
Spearman Rho tests the model's ability to actually rank images.  I took a small part of the validation dataset and ran more pairwise comparisons on them until they could be perfectly ranked from best to worst.  The model does the same, and those ranked lists are compared.  This is a useful metrics for saying "Okay, the model only has X accuracy on head-to-head comparisons, but how well does it do at overall ranking, even if individual pairs aren't perfect?"
Kendall Tau does a similar ranking-vs-ranking test, but pays more attention to the local structure of the ranking.
ECE is an odd metric, but generally should be low.  If it grows it means the model is becoming overconfident.
The rest of the metrics ... I dunno good interpretations of them honestly.  I found the dec_* metrics to be useless.  Brier is kind of a better accuracy measure, but I didn't look too closely at it.
Overall I just liked to see NLL going down, ECE low, Accuracy high, and Spearman high.
