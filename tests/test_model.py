"""Tests for the model and for batching.  Run:  uv run pytest

Most of these use a tiny model on the CPU, so they are fast and exact. The ones that matter most attack the
one way a language model can cheat: seeing the token it is supposed to predict.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gp_thee.data import DATA, load_tokens, load_works, random_batch
from gp_thee.model import GPT, Attention, Config
from gp_thee.tokenizer import load

TINY = dict(vocab_size=50, context=32, layers=2, heads=4, width=64)
on_gpu = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")


def tiny(**changes) -> GPT:
    torch.manual_seed(0)
    return GPT(Config(**{**TINY, "dropout": 0.0, **changes}))


# ------------------------------------------------------------------------------------------ size
@pytest.mark.parametrize("vocab_size, expected", [(98, 10_757_760), (1024, 11_113_344), (2048, 11_506_560), (4096, 12_292_992)])
def test_the_model_has_exactly_the_parameters_the_arithmetic_predicts(vocab_size, expected):
    model = GPT(Config(vocab_size=vocab_size))
    assert sum(p.numel() for p in model.parameters()) == expected == model.config.parameter_count()


def test_the_table_going_in_is_the_table_coming_out():
    model = tiny()
    assert model.to_scores.weight is model.token_embedding.weight  # one tensor, not two equal ones


def test_heads_must_divide_the_width():
    with pytest.raises(ValueError):
        Config(vocab_size=50, width=64, heads=5)


# ------------------------------------------------------------------------------------------ the loss
@pytest.mark.parametrize("vocab_size", [98, 2048])
def test_a_new_model_knows_nothing(vocab_size):
    # Before training, every token should be about equally likely, so the loss should be close to ln(vocab).
    torch.manual_seed(0)
    model = GPT(Config(vocab_size=vocab_size)).eval()
    tokens = torch.randint(0, vocab_size, (4, 256))
    _, loss = model(tokens, tokens.roll(-1, dims=1))
    assert math.log(vocab_size) <= loss.item() < math.log(vocab_size) + 0.15


def test_the_loss_is_the_average_surprise_at_the_right_answer():
    model = tiny().eval()
    tokens, targets = torch.randint(0, 50, (3, 32)), torch.randint(0, 50, (3, 32))
    scores, loss = model(tokens, targets)
    surprise = -F.log_softmax(scores, dim=-1).gather(2, targets.unsqueeze(2)).squeeze(2)  # nats, one per position
    assert torch.isclose(loss, surprise.mean(), atol=1e-6)


# ------------------------------------------------------------------------------------------ no peeking
@pytest.mark.parametrize("builtin", [True, False])
def test_the_future_cannot_change_the_past(builtin):
    # For every cut point k: scramble everything from position k onwards. The scores before k must not move
    # AT ALL. Not "by little": exactly zero. A masked share is exactly zero, and the scrambled text has the
    # same shape, so every sum is added up in the same order. (Compare texts of DIFFERENT lengths and the
    # last digits do move, harmlessly: see the short-text test below.)
    model = tiny(builtin_attention=builtin).eval()
    tokens = torch.randint(0, 50, (2, 32))
    with torch.no_grad():
        before, _ = model(tokens)
        for k in range(1, 32):
            scrambled = tokens.clone()
            scrambled[:, k:] = torch.randint(0, 50, (2, 32 - k))
            after, _ = model(scrambled)
            assert torch.equal(before[:, :k], after[:, :k]), f"position {k} or later leaked into the past"
            assert not torch.equal(before[:, k:], after[:, k:])  # and the test can see a change when there is one


@pytest.mark.parametrize("builtin", [True, False])
def test_no_gradient_flows_from_a_prediction_to_a_later_token(builtin):
    # The same promise, checked the way training sees it: the score at position t must have zero
    # derivative with respect to the input at any position after t.
    model = tiny(builtin_attention=builtin).eval()
    embedded = model.token_embedding(torch.randint(0, 50, (1, 32))).detach().requires_grad_(True)
    x = embedded + model.position_embedding(torch.arange(32))
    for block in model.blocks:
        x = block(x)
    for t in (0, 7, 30):
        (gradient,) = torch.autograd.grad(x[0, t].sum(), embedded, retain_graph=True)
        assert gradient[0, t + 1:].abs().max() == 0 and gradient[0, :t + 1].abs().max() > 0


def test_the_hand_written_attention_agrees_with_the_built_in_one():
    built_in, by_hand = tiny(builtin_attention=True).eval(), tiny(builtin_attention=False).eval()
    by_hand.load_state_dict(built_in.state_dict())
    tokens = torch.randint(0, 50, (4, 32))
    with torch.no_grad():
        assert torch.allclose(built_in(tokens)[0], by_hand(tokens)[0], atol=1e-5)


@pytest.mark.parametrize("builtin", [True, False])
def test_attention_matches_its_definition_worked_one_position_at_a_time(builtin):
    # The two attention paths share the code that splits the numbers into heads and puts them back, so comparing
    # them with each other cannot see a fault there. This oracle shares nothing: plain loops over rows, heads and
    # positions, on a text SHORTER than the context.
    torch.manual_seed(0)
    config = Config(**{**TINY, "dropout": 0.0, "builtin_attention": builtin})
    attention, x = Attention(config).eval(), torch.randn(3, 20, 64)
    size = config.width // config.heads
    with torch.no_grad():
        q, k, v = (x @ attention.query_key_value.weight.T).split(config.width, dim=2)
        mixed = torch.zeros_like(x)
        for row in range(3):
            for head in range(config.heads):
                mine = slice(head * size, (head + 1) * size)
                for t in range(20):
                    match = k[row, :t + 1, mine] @ q[row, t, mine] / math.sqrt(size)  # itself and everything before it
                    mixed[row, t, mine] = torch.softmax(match, dim=0) @ v[row, :t + 1, mine]
        assert torch.allclose(attention(x), mixed @ attention.output.weight.T, atol=1e-5)


@pytest.mark.parametrize("builtin", [True, False])
def test_a_short_text_gets_the_scores_it_would_get_as_the_start_of_a_long_one(builtin):
    # Training always uses full windows, but prompts are short. Nothing about the model may depend on the length.
    model = tiny(builtin_attention=builtin).eval()
    tokens = torch.randint(0, 50, (2, 32))
    with torch.no_grad():
        full, _ = model(tokens)
        for length in (1, 7, 20):
            assert torch.allclose(model(tokens[:, :length])[0], full[:, :length], atol=1e-5), length


def test_sequences_in_a_batch_do_not_affect_each_other():
    model = tiny().eval()
    tokens = torch.randint(0, 50, (4, 32))
    with torch.no_grad():
        together, _ = model(tokens)
        alone, _ = model(tokens[2:3])
    assert torch.allclose(together[2:3], alone, atol=1e-6)


# ------------------------------------------------------------------------------------------ training behaviour
@pytest.mark.parametrize("builtin", [True, False])
@pytest.mark.parametrize("site", ["embedding", "attention shares", "attention output", "feed-forward"])
def test_each_dropout_is_on_in_training_and_off_in_evaluation(builtin, site):
    # Four places drop values. Switch off all but one, and that one alone must make training passes differ,
    # while evaluation never differs. (One test for "is anything random?" would let three of them be broken.)
    model = tiny(dropout=0.5, builtin_attention=builtin)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            keep = {"embedding": name == "embedding_dropout", "attention output": name.endswith("attention.output_dropout"),
                    "feed-forward": name.endswith("feed_forward.dropout"), "attention shares": False}[site]
            module.p = 0.5 if keep else 0.0
        if isinstance(module, Attention):
            module.dropout = 0.5 if site == "attention shares" else 0.0
    tokens = torch.randint(0, 50, (2, 32))
    model.train()
    assert not torch.equal(model(tokens)[0], model(tokens)[0]), f"{site} dropout does nothing in training"
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(tokens)[0], model(tokens)[0]), f"{site} dropout is still on in evaluation"


def test_every_parameter_receives_a_gradient():
    model = tiny()
    tokens = torch.randint(0, 50, (2, 32))
    model(tokens, tokens.roll(-1, dims=1))[1].backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_every_position_has_its_own_row_and_uses_it():
    model = tiny()
    tokens = torch.full((1, 32), 7)  # the same token everywhere: only the position can tell the places apart
    scores, loss = model(tokens, tokens)
    assert all(not torch.allclose(scores[0, 0], scores[0, t], atol=1e-4) for t in range(1, 32))
    loss.backward()
    assert (model.position_embedding.weight.grad.abs().sum(dim=1) > 0).all()  # every row, not just the table as a whole


def test_a_block_is_the_two_lines_in_the_diagram():
    block, x = tiny().blocks[0].eval(), torch.randn(2, 32, 64)
    with torch.no_grad():
        step_1 = x + block.attention(block.norm_1(x))
        assert torch.allclose(block(x), step_1 + block.feed_forward(block.norm_2(step_1)), atol=1e-6)


def test_the_bend_in_the_feed_forward_is_a_gelu():
    feed_forward, x = tiny().blocks[0].feed_forward.eval(), torch.randn(2, 5, 64)
    with torch.no_grad():
        hidden = x @ feed_forward.expand.weight.T
        bent = hidden * 0.5 * (1 + torch.erf(hidden / math.sqrt(2)))  # GELU from its definition
        assert torch.allclose(feed_forward(x), bent @ feed_forward.shrink.weight.T, atol=1e-6)


def test_starting_values_are_the_ones_the_comment_describes():
    torch.manual_seed(0)
    model = GPT(Config(vocab_size=98))
    for name, p in model.named_parameters():
        if p.dim() < 2:
            assert torch.equal(p, torch.ones_like(p)), name               # every norm scale starts at exactly 1
        else:
            writes_to_x = name.endswith(("attention.output.weight", "feed_forward.shrink.weight"))
            expected = 0.02 / math.sqrt(2 * 6) if writes_to_x else 0.02
            assert abs(p.std().item() / expected - 1) < 0.05, (name, p.std().item())


def test_matrices_decay_and_norm_scales_do_not():
    model = GPT(Config(vocab_size=98))
    decayed, not_decayed = model.parameter_groups(0.1)
    assert (decayed["weight_decay"], not_decayed["weight_decay"]) == (0.1, 0.0)
    assert sum(p.numel() for p in decayed["params"]) == 10_752_768 and sum(p.numel() for p in not_decayed["params"]) == 4_992
    torch.optim.AdamW(model.parameter_groups(0.1))  # the tied table is in one group only, or this raises


def test_it_can_memorise_one_batch():
    # The classic end-to-end check: if the model, the loss and the optimizer are wired together correctly,
    # a model must be able to learn one small batch by heart. If it cannot, something is broken.
    model = tiny()
    tokens = torch.randint(0, 50, (4, 32))
    targets = tokens.roll(-1, dims=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = model(tokens, targets)[1].item()
    for _ in range(300):
        optimizer.zero_grad()
        loss = model(tokens, targets)[1]
        loss.backward()
        optimizer.step()
    assert first > 3.5 and loss.item() < 0.05


def test_text_longer_than_the_context_is_refused():
    with pytest.raises(ValueError, match="longer than the context"):
        tiny()(torch.zeros(1, 33, dtype=torch.long))


# ------------------------------------------------------------------------------------------ batches
def test_targets_are_the_tokens_moved_along_by_one():
    stream = load_tokens("char", "train")
    tokens, targets = random_batch(stream, 16, 64, np.random.default_rng(0), "cpu")
    assert tokens.shape == targets.shape == (16, 64) and tokens.dtype == torch.int64
    assert torch.equal(tokens[:, 1:], targets[:, :-1])
    begins = np.random.default_rng(0).integers(0, len(stream) - 64, size=16)  # the same draws the batch made
    for row, begin in enumerate(begins):  # every window is a real, unbroken stretch of the stream
        assert tokens[row].tolist() == stream[begin:begin + 64].tolist()
        assert targets[row].tolist() == stream[begin + 1:begin + 65].tolist()


def test_the_last_token_of_the_stream_can_be_a_target():
    stream = np.arange(40, dtype=np.uint16)
    last_targets = {int(random_batch(stream, 1, 8, np.random.default_rng(seed), "cpu")[1][0, -1]) for seed in range(300)}
    assert max(last_targets) == 39 and min(last_targets) == 8


@pytest.mark.parametrize("which", ["train", "validation"])
def test_the_saved_stream_is_the_works_it_claims_to_be(which):
    # Without this, a loader that handed back the validation stream as "train" passed every other test.
    tokenizer = load(DATA / "tokenizers" / "char.json")
    assert np.array_equal(load_tokens("char", which), np.array(tokenizer.encode_works(load_works(which))))
    report = json.loads((DATA / "tokenizers" / "report.json").read_text(encoding="utf-8"))
    for name, facts in report["tokenizers"].items():
        stream = load_tokens(name, which)
        assert len(stream) == facts["tokens"][which] and int(stream.max()) < facts["vocab_size"], name


def test_only_the_loader_opens_the_works():
    # The lock in data.py cannot stop a script that opens the files itself. So no code here may.
    root = Path(__file__).resolve().parent.parent
    allowed = {"prepare_data.py", "make_split.py", "data.py"}  # they write the works, define the split, and ARE the loader
    for path in [*root.glob("src/**/*.py"), *root.glob("scripts/*.py"), *root.glob("tests/*.py")]:
        if path.name not in allowed and path.name != Path(__file__).name:
            assert not re.search(r"processed.{0,6}works|works/\*|\.glob\(.{0,3}\*\.txt", path.read_text(encoding="utf-8")), path.name


def test_there_is_no_token_stream_for_the_test_works():
    with pytest.raises(PermissionError):
        load_tokens("char", "test")


# ------------------------------------------------------------------------------------------ the GPU
@on_gpu
@pytest.mark.parametrize("builtin", [True, False])
def test_the_gpu_agrees_with_the_cpu_forwards_and_backwards(builtin):
    cpu, tokens = tiny(builtin_attention=builtin), torch.randint(0, 50, (4, 32))
    gpu = tiny(builtin_attention=builtin).to("mps")
    targets = tokens.roll(-1, dims=1)
    scores_cpu, loss_cpu = cpu(tokens, targets)
    scores_gpu, loss_gpu = gpu(tokens.to("mps"), targets.to("mps"))
    loss_cpu.backward()
    loss_gpu.backward()
    assert torch.allclose(scores_cpu, scores_gpu.cpu(), atol=1e-4)
    for (name, p), q in zip(cpu.named_parameters(), gpu.parameters()):
        assert torch.allclose(p.grad, q.grad.cpu(), atol=1e-5), name


@on_gpu
@pytest.mark.parametrize("builtin", [True, False])
@pytest.mark.parametrize("sixteen_bit", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_the_future_cannot_change_the_past_on_the_gpu(builtin, sixteen_bit, training):
    # With gradients ON, PyTorch's built-in attention takes a different route through the GPU than with them
    # off, and 16-bit takes another. Training uses the first; evaluation and sampling use the second. Check all.
    model = tiny(builtin_attention=builtin).to("mps").train(training)  # dropout is 0, so training mode is still exact
    tokens = torch.randint(0, 50, (2, 32))
    with torch.set_grad_enabled(training), torch.autocast("mps", dtype=torch.bfloat16, enabled=sixteen_bit):
        before = model(tokens.to("mps"))[0].detach().float().cpu()
        for k in range(1, 32):
            scrambled = tokens.clone()
            scrambled[:, k:] = torch.randint(0, 50, (2, 32 - k))
            after = model(scrambled.to("mps"))[0].detach().float().cpu()
            assert torch.equal(before[:, :k], after[:, :k]), f"leak across position {k}"
