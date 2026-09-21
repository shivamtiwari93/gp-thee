# Plan

## Goals

1. Build the model and document every step in detail.
2. Publish it on GitHub and Hugging Face as an educational project.
3. Write a detailed blog post about it.
4. Make a 5-slide PDF and a post for LinkedIn.

Goal 1 produces the material for the other three. The record of it is [BUILD_LOG.md](BUILD_LOG.md).

## The rule

Shakespeare is the only text in the universe. That bans pretrained weights, pretrained tokenizers (for example GPT-2's vocabulary, or `tiktoken`) and any outside text. Tools are allowed as algorithms, not as sources of knowledge: PyTorch is fine, a vocabulary learned from the web is not.

## Phase 1: build

| Step | What | Done when |
|---|---|---|
| 1. Environment | uv-managed Python 3.14, PyTorch on the Mac GPU (MPS) | a tensor multiplies on `mps` |
| 2. Data | one verified download, deterministic cleaning script with assertions | cleaned corpus committed with its checksum and a per-work manifest |
| 3. Split | hold out whole works: about 90% train, 5% validation, 5% test | the list of works is frozen and committed before any training |
| 4. Tokenizer | character-level first, then a BPE learned from the training split only (sizes 1024, 1536, 2048, 4096) | `decode(encode(text)) == text` on the whole corpus |
| 5. Model | nanoGPT-style decoder: 6 layers, 6 heads, width 384, context 256; hand-written attention, with PyTorch's built-in one as a cross-check | parameter count matches the hand calculation; predictions match an independent NumPy reference |
| 6. Correctness checks | initial loss near ln(vocab), overfit one batch, causal-leak test, CPU vs GPU parity | all pass |
| 7. Benchmark | 2-minute throughput test, fp32 and bf16 | tokens per second recorded; run budgets set from it |
| 8. Train | 32-bit, hand-written attention. AdamW; matrices decay, norm scales do not; warm-up then cosine, stated in PASSES over the training stream (not steps: 5,000 steps is 17 passes of characters but 42 to 54 of BPE); gradients clipped at 1.0; best checkpoint on validation kept. Re-run the gate's no-peeking check on the first TRAINED model (every gate check so far ran on a model that knows nothing). Validation = one deterministic 32-bit pass over the whole stream with overlapping windows, summed nats. Same-seed runs are not bit-identical on this GPU, so report that spread too. **Done** (BUILD_LOG entry 14): run length per tokenizer by a doubling rule (characters: 34 passes), 40 evaluations per run, seeds 1 to 3 for comparisons | loss curves saved |
| 9. Evaluate | bits per character = total loss in nats over every target, divided by ln 2 and by the set's character count, so tokenizers are comparable. Reported per work, and separately for speaker-label lines (unseen names dominate them). Baselines, fitted on the training works and scored on validation: character n-grams of orders 1 to 8 with interpolated Witten-Bell smoothing (no constant to tune; the best is the 6-gram at 2.231), bzip2 and xz alone and after reading the training works. **Done for validation**; the test works wait for the final evaluation | the model beats the n-grams and the compressors (it does, by 0.46 and more) |
| 10. Memorisation | share of 50-character windows copied verbatim from training text | reported per checkpoint and temperature |
| 11. Sample | `sample.py`, and a "speak as a character" dialogue wrapper. Prompts pass through a normaliser outside the tokenizer: straight quotes to curly, tabs and carriage returns handled, trailing spaces stripped (a space belongs to the NEXT token), anything else refused. Never prepend START to an ordinary prompt | fixed 10-prompt suite regenerated per checkpoint |
| 12. Experiments | **Tokenizer: done (BUILD_LOG entry 16): characters, 1.7546 against 1.8012 to 1.8511 for the four vocabularies.** Tokenizer (chosen by validation bits per character over all the text, mean of 3 seeds, each run at its best validation checkpoint and at its own run length; "the spread" is the standard deviation over seeds pooled over all arms, two arms differ only beyond 1.82 of it, and within it the smaller vocabulary wins: BUILD_LOG entry 14), model size (about 5M / 11M / 38M, same tokenizer and context), dropout, one variable at a time, 3 seeds each | results table |

Optional later: a port to Apple's MLX, modern architecture tweaks (RoPE, RMSNorm, SwiGLU), an ensemble.

## Phase 2: publish

- GitHub: this repo, public from day one.
- Hugging Face: weights as safetensors (never a `.pt`, which is a program; the tied output table is dropped on export and re-tied on load, and the loaded model must reproduce the checkpoint's validation score), tokenizer files, a model card that states plainly what the model can and cannot do, a loader script. Possibly a small live demo.

## Phase 3: blog post

Written from the build log: the premise, each step, the mistakes, the charts, sample output at different stages of training.

## Phase 4: LinkedIn

A 5-slide PDF and a post, using the real charts and samples.

## Naming

The repo is `gp-thee`. Each released checkpoint carries its parameter count: the first is `GP-Thee-11M` (10,757,760 parameters with the character tokenizer: 97 characters plus START). The count is fixed by the architecture, so it is known before training; see the build log.
