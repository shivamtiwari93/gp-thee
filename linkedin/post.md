# LinkedIn post

*Paste the text below. Attach `gp-thee-carousel.pdf` as a document/carousel (LinkedIn → "Add a document"). A short first line is the hook that shows before "…see more".*

---

I trained a GPT from scratch on nothing but Shakespeare. The hardest part wasn't the model.

For a small side project, I gave a language model one rule: the only text that has ever existed is Shakespeare. 38 plays, 154 sonnets, 5 poems — about 5 MB. No pretraining, no borrowed tokenizer, no outside text. I built every piece from scratch — the tokenizer, the 11-million-parameter model, the training loop — and trained it on a laptop. It's called GP-Thee-11M.

Getting it to write convincing pseudo-Shakespeare was the easy half. The hard half was answering one question *honestly*: how good is it, really?

The trap in machine learning is that it's very easy to fool yourself. So the whole project was built around not doing that:

→ On day one, before any training, I locked away three works — a history, a romance, a poem — and the code physically refused to open them.

→ Every rule was written down *before* the number it governed: which model to release, what would count as "memorising," how to score it. No decision was ever made after seeing the result.

→ The final test could be run exactly once. A held-out score is only held out until you use it. So I built a one-way door: the moment those three works were read, the measurement was spent forever.

The result, on Shakespeare it had never seen:

**1.80 bits per character** — it beats every predictor I could build without a neural network (the best n-gram: 2.32; a good file compressor: 2.6). From 5 MB and 11M parameters, no pretraining.

And my favourite finding: it has essentially *not* memorised the plays. The longest passage of Shakespeare's actual verse it can be made to recite is 25 characters. What it learned by heart was the *shape* of an edition — the cast lists, the scene headings, the speaker labels — not the poetry.

It's an in-character autocomplete for a world made only of Shakespeare. It can't answer a question or follow an instruction. Give it a speaker and a line and it plays the next part, and nothing else.

I built it for one reason: to see whether a language model could be made, and measured, honestly — start to finish, in the open, with every number checked before it was written. It could.

Everything is public: the code, every training log, the weights, and an eight-part write-up of every step (including the mistakes, which there were several of, all corrected in the open).

🔗 Code & write-up: github.com/shivamtiwari93/gp-thee
🤖 Weights: huggingface.co/shivamtiwari93/gp-thee-11m

#MachineLearning #LLM #DeepLearning #AI #NLP #Shakespeare #BuildInPublic
