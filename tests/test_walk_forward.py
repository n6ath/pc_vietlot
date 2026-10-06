import unittest
from pathlib import Path

from analysis.walk_forward import (
    FEATURE_WARMUP_DRAWS,
    INITIAL_TRAINING_DRAWS,
    _feature_rows,
    _score,
    _training_matrices,
    _load_draws,
)
from analyzer import BASE_DIR, GAME_CONFIGS


class WalkForwardTests(unittest.TestCase):
    def test_training_matrices_stop_before_target_draw(self):
        game_type = "5_35"
        _, max_number, _, _ = GAME_CONFIGS[game_type]
        draws = _load_draws(game_type, BASE_DIR / "data")
        features, labels = _training_matrices(
            draws, max_number, INITIAL_TRAINING_DRAWS
        )

        self.assertEqual(
            len(features), INITIAL_TRAINING_DRAWS - FEATURE_WARMUP_DRAWS
        )
        self.assertEqual(len(labels), len(features))
        expected_labels = [
            [
                int(number in draws[index]["numbers"])
                for number in range(1, max_number + 1)
            ]
            for index in range(FEATURE_WARMUP_DRAWS, INITIAL_TRAINING_DRAWS)
        ]
        self.assertEqual(labels, expected_labels)

    def test_target_features_do_not_change_when_future_draws_are_added(self):
        game_type = "6_45"
        _, max_number, _, _ = GAME_CONFIGS[game_type]
        draws = _load_draws(game_type, BASE_DIR / "data")
        target_index = INITIAL_TRAINING_DRAWS
        prefix_features = _feature_rows(draws[:target_index], max_number)[-1]
        full_features = _feature_rows(draws, max_number)[target_index]

        self.assertEqual(prefix_features, full_features)

    def test_hit_count_uses_main_numbers(self):
        game_type = "6_55"
        draws = _load_draws(game_type, BASE_DIR / "data")
        actual = draws[INITIAL_TRAINING_DRAWS]["numbers"]

        self.assertEqual(_score(actual, actual), len(actual))
        self.assertEqual(_score(actual[:3], actual[3:]), 0)

    def test_loaded_history_is_chronological_and_has_warmup(self):
        for game_type in GAME_CONFIGS:
            with self.subTest(game=game_type):
                draws = _load_draws(game_type, Path(BASE_DIR / "data"))
                self.assertGreaterEqual(len(draws), INITIAL_TRAINING_DRAWS)
                self.assertEqual(
                    [draw["draw_datetime"] for draw in draws],
                    sorted(draw["draw_datetime"] for draw in draws),
                )


if __name__ == "__main__":
    unittest.main()
