"""
Unit tests for the full-parameter SFT training helpers (F1.4 / F2.3).

Only the pure label-masking logic is tested here; the training loop itself
requires a CUDA GPU and runs on the cloud host.
"""
import os
import sys

_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(_SRC_DIR))

from full_sft import IGNORE_INDEX, prompt_masked_labels  # noqa: E402


def test_prompt_masked_labels_masks_prompt_only():
    input_ids = [101, 102, 103, 201, 202]
    labels = prompt_masked_labels(input_ids, prompt_len=3)
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 201, 202]


def test_prompt_masked_labels_keeps_all_when_prompt_is_empty():
    input_ids = [1, 2, 3]
    assert prompt_masked_labels(input_ids, prompt_len=0) == input_ids


def test_prompt_masked_labels_clamps_long_prompt():
    input_ids = [1, 2, 3]
    assert prompt_masked_labels(input_ids, prompt_len=99) == [IGNORE_INDEX] * 3


def test_prompt_masked_labels_clamps_negative_prompt():
    input_ids = [1, 2, 3]
    assert prompt_masked_labels(input_ids, prompt_len=-5) == input_ids


def test_prompt_masked_labels_supports_custom_ignore_index():
    assert prompt_masked_labels([1, 2], 1, ignore_index=-1) == [-1, 2]


def test_prompt_masked_labels_empty_input():
    assert prompt_masked_labels([], 0) == []
