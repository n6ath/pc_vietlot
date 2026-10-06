import unittest

from analysis.walk_forward_benchmark import (
    _holm_adjust,
    _hypergeometric_pmf,
    _rank_frequency,
    _rank_gap,
    _random_total_distribution,
    _summarize_method,
    _two_sided_exact_p,
)


class WalkForwardBenchmarkTests(unittest.TestCase):
    def test_random_hit_pmf_has_hypergeometric_probabilities(self):
        pmf = _hypergeometric_pmf(5, 2)

        self.assertAlmostEqual(sum(pmf), 1.0)
        self.assertEqual(len(pmf), 3)
        self.assertAlmostEqual(pmf[0], 0.3)
        self.assertAlmostEqual(pmf[1], 0.6)
        self.assertAlmostEqual(pmf[2], 0.1)

    def test_random_total_distribution_and_exact_test(self):
        distribution = _random_total_distribution(2, 5, 2)

        self.assertAlmostEqual(sum(distribution), 1.0)
        self.assertAlmostEqual(
            sum(index * probability for index, probability in enumerate(distribution)),
            1.6,
        )
        self.assertAlmostEqual(_two_sided_exact_p(0, distribution), 0.1)
        self.assertAlmostEqual(_two_sided_exact_p(2, distribution), 1.0)

    def test_holm_adjustment_preserves_monotonic_order(self):
        adjusted = _holm_adjust([0.01, 0.04, 0.03])

        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_summary_compares_total_hits_to_random_hit_mean(self):
        distribution = _random_total_distribution(2, 5, 2)
        summary = _summarize_method([1, 2], 2, 0.8, distribution)

        self.assertEqual(summary["cumulative_total_hits"], 3)
        self.assertAlmostEqual(
            summary["random_baseline_expected_total_hits"], 1.6
        )
        self.assertAlmostEqual(
            summary["difference_from_random_expected_total"], 1.4
        )

    def test_baselines_only_use_supplied_history(self):
        history = [
            {"numbers": [1, 2]},
            {"numbers": [2, 3]},
            {"numbers": [3, 4]},
        ]

        self.assertEqual(_rank_frequency(history, 5, 2), [2, 3])
        self.assertEqual(_rank_gap(history, 5, 2), [5, 1])


if __name__ == "__main__":
    unittest.main()
