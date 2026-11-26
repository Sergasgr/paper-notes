# paper-notes

My AI/ML paper summaries.

## Note Structure

```
# <Title> (<year>, <lab or authors>)
**Guiding Question:**
**Problem:** what didn't work before, in 1–2 sentences
**Key Idea:** one sentence understandable to someone in my class
**Method:** the central equation or diagram, written by me
**Key Result:** the important number and against which baseline (+ my experiments, if any)
**Limitations:** what they don't prove, assumptions, cost
**Connection:** tags + relationship to TFG / CodeAlign / Distill / peft / exam question
**Questions:** 3 interview-style (conceptual, math/design, critique/trade-offs), with answers
**Flashcards:** 3–5, in Q&A format
```

## Folders

```
eng/        notes in English
esp/        notes in Spanish
  peft/     PEFT methods (+ my experiments on the peft image-gen benchmark)
  rl/
  ...
```

## Experiments on `FLUX.2-klein`

The notes in `peft/` include my runs on the image-gen method comparison benchmark from [huggingface/peft](https://github.com/huggingface/peft) (`FLUX.2-klein`, *few-shot* subject with 2 validation images and 4 test images, 750 steps):

- **DINOv2 similarity (test):** DINOv2 cosine similarity with the test images (subject fidelity; higher = better).
- **Drift:** how much images change for unrelated prompts (e.g., *"an astronaut riding a horse"*) compared to the base model with the same seed: (1 − mean DINOv2 cosine similarity) / 2; 0 = identical images.
- **Baseline:** LoRA r=32 (DINOv2 0.6815 / drift 0.2721 on my RTX 5070 Ti; 0.6972 / 0.2620 on the official benchmark's L40S).
