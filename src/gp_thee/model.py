"""The model: a GPT, small enough to read in one sitting.

Text goes in as token ids and comes out as one score per vocabulary piece for what the NEXT token is, at
every position at once. In between:

    token ids
       │  look up a row of `width` numbers for each token, add a row for its position
       ▼
    ┌─ block, repeated `layers` times ───────────────────────────────────────────────────────┐
    │  x = x + attention(norm(x))     every position gathers information from itself and     │
    │                                 from earlier positions                                 │
    │  x = x + feed_forward(norm(x))  every position thinks about what it gathered, alone     │
    └─────────────────────────────────────────────────────────────────────────────────────────┘
       │  one last norm
       ▼
    scores for the next token   (computed with the SAME table that looked the tokens up)

Two things in that picture do most of the work and are easy to miss:

  * `x = x + ...`  Each step ADDS to x instead of replacing it. These are the residual connections. They
    give every layer a direct line to the input and to the output, which is what makes a deep stack of
    layers trainable at all.
  * Attention may only look BACKWARDS. If position 5 could see token 6, predicting token 6 would be
    copying. The training loss would fall towards zero while the model learned only to copy, and when it
    came to writing new text there would be nothing to copy from. The causal mask enforces this, and the
    tests attack it harder than anything else in this file.

This is the GPT-2 arrangement (a norm BEFORE each step), with no bias terms anywhere.

If you know nanoGPT or the GPT-2 paper, the plain-English names here map onto theirs like this:
    width = n_embd / d_model        context = block_size        layers = n_layer        heads = n_head
    token_embedding = wte           position_embedding = wpe    to_scores = lm_head     scores = logits
    query_key_value = c_attn        Attention.output = c_proj   expand = mlp.c_fc       shrink = mlp.c_proj
    norm_1, norm_2, final_norm = ln_1, ln_2, ln_f               may_look = the lower-triangular mask ("bias")
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int            # how many different tokens exist (98 for the character tokenizer)
    context: int = 256         # how many tokens the model can look back over
    layers: int = 6            # how many blocks are stacked
    heads: int = 6             # attention heads per block; each works on width / heads of the numbers
    width: int = 384           # how many numbers represent each token
    dropout: float = 0.2       # share of values randomly zeroed during TRAINING, so no single path is relied on
                               # (the survivors are scaled up by 1 / (1 - dropout), so the average is unchanged)
    builtin_attention: bool = False
    # False: the attention written out below, line by line. The same code runs in training, evaluation and sampling.
    # True:  PyTorch's F.scaled_dot_product_attention. On Apple GPUs (PyTorch 2.14) that is a fast fused kernel
    #        only when gradients are OFF. During training it quietly runs the same separate steps as our version,
    #        so it is no faster and saves no memory here. It is kept as an independent cross-check.

    def __post_init__(self):
        if self.width % self.heads:
            raise ValueError(f"width {self.width} must divide evenly among {self.heads} heads")

    def parameter_count(self) -> int:
        """Known before anything is built: see blog part 1 for where each term comes from."""
        d, L, V, T = self.width, self.layers, self.vocab_size, self.context
        return 12 * d * d * L + (2 * L + 1) * d + V * d + T * d


class Attention(nn.Module):
    """Each position asks a question of itself and every EARLIER position, and takes a weighted mix of their answers.

    From its `width` numbers, every position makes three things: a query (what am I looking for?), a key
    (what do I have?), and a value (what I will hand over if asked). How well a query matches a key decides
    how much of that position's value gets mixed in. `heads` of these lookups run side by side, each on its
    own slice of the numbers, and one last matrix combines them.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.heads, self.dropout, self.builtin = config.heads, config.dropout, config.builtin_attention
        self.query_key_value = nn.Linear(config.width, 3 * config.width, bias=False)  # the three d x d matrices, stacked
        self.output = nn.Linear(config.width, config.width, bias=False)               # the fourth
        self.output_dropout = nn.Dropout(config.dropout)
        # True where looking is allowed: row t has True in columns 0..t. The diagonal matters: position 0 has nobody
        # else to look at, and a row with no True at all would come out of softmax as NaN. This is a buffer, not a
        # parameter, so it is never trained; persistent=False also keeps it out of saved checkpoints.
        self.register_buffer("may_look", torch.tril(torch.ones(config.context, config.context, dtype=torch.bool)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        query, key, value = self.query_key_value(x).split(width, dim=2)
        # (batch, length, width) -> (batch, heads, length, width / heads): each head gets its own slice.
        query, key, value = (t.view(batch, length, self.heads, width // self.heads).transpose(1, 2) for t in (query, key, value))

        if self.builtin:
            mixed = F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        else:
            # The same thing, spelled out. match[b, h, t, s] = how well position t's query matches position s's key.
            match = (query @ key.transpose(-2, -1)) / math.sqrt(key.shape[-1])  # divide so the numbers do not grow with head size
            match = match.masked_fill(~self.may_look[:length, :length], float("-inf"))  # the future gets minus infinity...
            shares = F.softmax(match, dim=-1)   # ...which softmax turns into a share of exactly zero. Each row sums to 1.
            shares = F.dropout(shares, self.dropout, self.training)  # (these "attention weights" are not parameters)
            mixed = shares @ value

        mixed = mixed.transpose(1, 2).contiguous().view(batch, length, width)  # put the heads back side by side
        return self.output_dropout(self.output(mixed))


class FeedForward(nn.Module):
    """Two matrices with a bend between them, applied to each position separately: width -> 4 x width -> width."""

    def __init__(self, config: Config):
        super().__init__()
        self.expand = nn.Linear(config.width, 4 * config.width, bias=False)
        self.shrink = nn.Linear(4 * config.width, config.width, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.shrink(F.gelu(self.expand(x))))  # GELU is the bend; without it, two matrices are just one matrix


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # A norm shifts each position's numbers to average 0 and rescales them to spread 1, then multiplies
        # each by a learned scale. It keeps the numbers in a sensible range however deep the stack.
        self.norm_1 = nn.LayerNorm(config.width, bias=False)
        self.attention = Attention(config)
        self.norm_2 = nn.LayerNorm(config.width, bias=False)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_1(x))
        x = x + self.feed_forward(self.norm_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context, config.width)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.width, bias=False)
        self.to_scores = nn.Linear(config.width, config.vocab_size, bias=False)
        self.to_scores.weight = self.token_embedding.weight  # weight tying: one table, used going in and coming out

        # Starting values. Small random numbers, so that at the start every token is about equally likely and
        # the first loss is close to ln(vocab_size). (On random tokens it sits about 0.02^2 x width / 2 = 0.08
        # above it: that is what scores with a spread of 0.02 x sqrt(width) cost.) The two matrices that write back into x in each block
        # start smaller still, by 1 / sqrt(2 x layers): there are 2 x layers of those additions, and this keeps
        # their sum from growing with depth. (This is GPT-2's recipe.)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and module is not self.to_scores:  # its table is token_embedding's
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            for writes_to_x in (block.attention.output, block.feed_forward.shrink):
                nn.init.normal_(writes_to_x.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.layers))

        counted = sum(p.numel() for p in self.parameters())
        if counted != config.parameter_count():
            raise AssertionError(f"built {counted:,} parameters, but the arithmetic says {config.parameter_count():,}")

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        """tokens: (batch, length) of ids. Returns scores (batch, length, vocab_size), and the loss if targets are given.

        targets[b, t] is the token that really came after tokens[b, :t + 1], so targets is tokens shifted left by one.
        The loss is the average surprise, in nats, over every position of every sequence.
        """
        batch, length = tokens.shape
        if length > self.config.context:
            raise ValueError(f"{length} tokens is longer than the context of {self.config.context}")
        positions = torch.arange(length, device=tokens.device)
        x = self.embedding_dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        scores = self.to_scores(self.final_norm(x))
        if targets is None:
            return scores, None
        # .float(): when the heavy arithmetic runs in 16-bit, the loss is still worked out in 32-bit.
        return scores, F.cross_entropy(scores.float().view(-1, scores.shape[-1]), targets.reshape(-1))

    def parameter_groups(self, weight_decay: float) -> list[dict]:
        """For the optimizer: every matrix and table decays, every norm scale does not.

        Weight decay pulls numbers towards zero. That is a sensible pull for a matrix, and the wrong one for a
        norm's scale, whose neutral value is 1. Grouping is by tensor, not by layer: the token table is ALSO
        the output matrix, and a tensor may only be in one group.
        """
        matrices = [p for p in self.parameters() if p.dim() >= 2]
        scales = [p for p in self.parameters() if p.dim() < 2]
        return [{"params": matrices, "weight_decay": weight_decay}, {"params": scales, "weight_decay": 0.0}]
