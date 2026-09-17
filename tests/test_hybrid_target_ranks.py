"""Regression tests for Hybrid Top-K target-rank extraction."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.recommendation.evaluation.loo_metrics import metrics_from_target_ranks

try:
    import torch
except ModuleNotFoundError:  # Local release checks may not include the GPU runtime.
    torch = None
else:
    from experiments.recommendation.runners.run_hybrid import _target_ranks


@unittest.skipIf(torch is None, "PyTorch is not installed in this local environment.")
class HybridTargetRanksTest(unittest.TestCase):
    def test_int64_item_indices_are_stored_as_int16_ranks(self) -> None:
        top_rows = torch.tensor(
            [[7, 4, 1], [9, 2, 5], [8, 6, 3], [10, 11, 12]],
            dtype=torch.int64,
        )
        target_rows = torch.tensor([7, 2, 0, 12], dtype=torch.int64)

        actual = _target_ranks(top_rows, target_rows)
        expected = np.array([1, 2, 0, 3], dtype=np.int16)

        self.assertEqual(top_rows.dtype, torch.int64)
        self.assertEqual(target_rows.dtype, torch.int64)
        self.assertEqual(actual.dtype, np.dtype("int16"))
        np.testing.assert_array_equal(actual, expected)

    def test_ranks_and_metrics_match_numpy_reference(self) -> None:
        top_rows_np = np.array(
            [[21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]],
            dtype=np.int64,
        )
        target_rows_np = np.array([24, 99, 41], dtype=np.int64)
        expected = np.array(
            [
                int(np.flatnonzero(row == target)[0]) + 1
                if np.any(row == target)
                else 0
                for row, target in zip(top_rows_np, target_rows_np, strict=True)
            ],
            dtype=np.int16,
        )

        actual = _target_ranks(
            torch.from_numpy(top_rows_np), torch.from_numpy(target_rows_np)
        )
        np.testing.assert_array_equal(actual, expected)

        topk = (1, 2, 4)
        actual_metrics = metrics_from_target_ranks(actual, topk)
        expected_metrics = metrics_from_target_ranks(expected, topk)
        self.assertEqual(actual_metrics, expected_metrics)


if __name__ == "__main__":
    unittest.main()
