"""Small CPU-only tests for the public release."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import binom

from watermark_attribution.metrics import (
    false_detection_rate,
    match_watermarks,
    maximum_pairwise_accuracy,
    true_detection_and_attribution_rates,
)
from watermark_attribution.selection import (
    _bounded_search_tree,
    select_watermarks,
)
from watermark_attribution.theory import (
    paper_table_5,
    tar_lower_bound,
    tdr_lower_bound,
)


class TheoryTests(unittest.TestCase):
    def test_table_5_values(self) -> None:
        bounds = paper_table_5()
        self.assertAlmostEqual(
            bounds["TDR lower bound"],
            0.9999962280431393,
            places=12,
        )
        self.assertAlmostEqual(
            bounds["FDR upper bound"],
            0.05998120543727691,
            places=12,
        )
        self.assertAlmostEqual(
            bounds["TAR lower bound"],
            0.9999962280431393,
            places=12,
        )

    def test_tdr_integer_boundaries(self) -> None:
        expected = binom.sf(6, 10, 0.9) + binom.cdf(1, 10, 0.9)
        actual = tdr_lower_bound(
            n=10,
            beta=0.9,
            tau=0.7,
            alpha_min=0.2,
        )
        self.assertAlmostEqual(actual, expected)

    def test_tar_uses_attribution_threshold(self) -> None:
        # max(floor((1 + 0.6) * 10 / 2) + 1, ceil(0.7 * 10)) = 9
        expected = binom.sf(8, 10, 0.9)
        actual = tar_lower_bound(
            n=10,
            beta=0.9,
            tau=0.7,
            alpha_max=0.6,
        )
        self.assertAlmostEqual(actual, expected)


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = np.array(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.uint8,
        )
        self.decoded = np.array(
            [
                [0, 0, 0, 1],
                [1, 1, 1, 1],
                [0, 1, 0, 1],
            ],
            dtype=np.uint8,
        )

    def test_matching_uses_inclusive_threshold(self) -> None:
        detected, predicted, scores = match_watermarks(
            self.decoded,
            self.pool,
            tau=0.75,
        )
        np.testing.assert_array_equal(detected, [True, True, False])
        np.testing.assert_array_equal(predicted[:2], [0, 1])
        self.assertEqual(scores[0], 0.75)

    def test_rates(self) -> None:
        targets = np.array([0, 1, 2])
        tdr, tar = true_detection_and_attribution_rates(
            self.decoded,
            targets,
            self.pool,
            tau=0.75,
        )
        self.assertAlmostEqual(tdr, 2 / 3)
        self.assertAlmostEqual(tar, 2 / 3)
        self.assertAlmostEqual(
            false_detection_rate(self.decoded, self.pool, tau=0.75),
            2 / 3,
        )

    def test_maximum_pairwise_accuracy(self) -> None:
        self.assertEqual(maximum_pairwise_accuracy(self.pool), 0.5)


class SelectionTests(unittest.TestCase):
    def test_methods_are_deterministic_binary_and_unique(self) -> None:
        for method in ("random", "nrg", "absta"):
            with self.subTest(method=method):
                first = select_watermarks(method, 12, n=16, seed=7)
                second = select_watermarks(method, 12, n=16, seed=7)
                np.testing.assert_array_equal(first, second)
                self.assertEqual(first.shape, (12, 16))
                self.assertTrue(np.all((first == 0) | (first == 1)))
                self.assertEqual(np.unique(first, axis=0).shape[0], 12)

    def test_bsta_respects_match_limit(self) -> None:
        existing = np.array(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        candidate = np.array([0, 0, 0, 1], dtype=np.uint8)
        result = _bounded_search_tree(
            existing,
            candidate,
            depth=1,
            max_matches=2,
            rng=np.random.default_rng(0),
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertLessEqual(
            int(np.max(np.count_nonzero(existing == result, axis=1))),
            2,
        )


if __name__ == "__main__":
    unittest.main()
