"""
Unit tests for GRPO batch planning.

Pure arithmetic, so the divisibility rule TRL enforces at runtime is checked
locally instead of failing after model loading on the GPU box.
"""
import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC_DIR))

from grpo import plan_generation_batch  # noqa: E402


def test_generation_batch_matches_trl_when_already_divisible():
    # batch 2 x accum 4 = 8, which TRL accepts for G=8.
    assert plan_generation_batch(8, 2, 4) == 8


def test_generation_batch_grows_to_cover_num_generations():
    # The failing configuration: batch 2 x accum 4 = 8 < G=16.
    assert plan_generation_batch(16, 2, 4) == 16
    assert plan_generation_batch(16, 2, 8) == 16


def test_generation_batch_result_is_always_a_multiple_of_num_generations():
    for num_generations in (4, 8, 16):
        for batch_size in (1, 2, 4):
            for accum in (1, 2, 4, 8):
                value = plan_generation_batch(num_generations, batch_size, accum)
                assert value % num_generations == 0
                assert value >= batch_size * accum
