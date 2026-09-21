"""The rule that chooses how long a tokenizer's model trains was written down in advance. These tests pin the program to the rule."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("find_length", Path(__file__).resolve().parent.parent / "scripts" / "find_length.py")
find_length = importlib.util.module_from_spec(spec)
spec.loader.exec_module(find_length)
next_to_try, choose = find_length.next_to_try, find_length.choose


def run(best, at, of):
    return {"best": best, "best_step": round(at * of)}


def test_it_starts_near_5000_steps_and_always_tries_double():
    assert next_to_try({}) == 5_000
    assert next_to_try({5_000: run(1.80, 1.0, 5_000)}) == 10_000
    assert next_to_try({5_000: run(1.80, 0.3, 5_000)}) == 10_000            # even a run that over-fitted early: we cannot know without trying


def test_it_doubles_while_doubling_gains_more_than_a_hundredth_and_stops_at_20000():
    tried = {5_000: run(1.800, 1.0, 5_000), 10_000: run(1.780, 0.98, 10_000)}
    assert next_to_try(tried) == 20_000                                     # 0.020 gained: double again
    tried[20_000] = run(1.775, 0.97, 20_000)
    assert next_to_try(tried) is None                                       # 0.005 gained: stop. Nothing over-fitted early, so no halves either
    assert choose(tried) == 10_000                                          # 20,000 is the lowest, 10,000 is within 0.01 of it, and shorter
    tried[20_000] = run(1.700, 0.97, 20_000)
    assert next_to_try(tried) is None and choose(tried) == 20_000           # still gaining, but 20,000 is the cap
    assert next_to_try({5_000: run(1.800, 1.0, 5_000), 10_000: run(1.7901, 1.0, 10_000)}) is None   # 0.0099 is not more than 0.01


def test_a_run_whose_best_moment_comes_early_sends_it_to_try_half_and_the_half_too():
    tried = {5_000: run(1.900, 0.30, 5_000), 10_000: run(1.930, 0.14, 10_000)}
    assert next_to_try(tried) == 2_500                                      # 10,000's half is 5,000, already made; 5,000's half is next
    tried[2_500] = run(1.880, 0.52, 2_500)
    assert next_to_try(tried) == 1_250                                      # 52% is before two thirds
    tried[1_250] = run(1.885, 0.96, 1_250)
    assert next_to_try(tried) is None                                       # its best moment is late: nothing more to try
    assert choose(tried) == 1_250                                           # the lowest is 2,500's 1.880; 1,250 is within 0.01 and shorter
    tried[1_250] = run(1.895, 0.96, 1_250)
    assert choose(tried) == 2_500                                           # 0.015 behind is not within 0.01


def test_two_thirds_is_the_line_and_625_steps_is_the_floor():
    assert next_to_try({5_000: run(1.9, 0.66, 5_000), 10_000: run(1.95, 0.9, 10_000)}) == 2_500
    assert next_to_try({5_000: run(1.9, 0.67, 5_000), 10_000: run(1.95, 0.9, 10_000)}) is None
    tried = {steps: run(1.9, 0.1, steps) for steps in (10_000, 5_000, 2_500, 1_250)}
    assert next_to_try(tried) == 625
    tried[625] = run(1.9, 0.1, 625)
    assert next_to_try(tried) is None                                       # 312 steps would be under the floor
