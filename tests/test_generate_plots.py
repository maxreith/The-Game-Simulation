"""Tests for generate_plots module."""

import pandas as pd

from generate_plots import find_optimal_parameters


class TestFindOptimalParameters:
    """Tests for find_optimal_parameters function."""

    def test_finds_row_with_highest_win_rate(self):
        df = pd.DataFrame(
            {
                "n_players": [2, 3, 4],
                "bonus_play_threshold": [1, 2, 3],
                "win_rate": [0.1, 0.5, 0.3],
            }
        )
        result = find_optimal_parameters(df)
        assert result == {"n_players": 3, "bonus_play_threshold": 2}

    def test_handles_single_row(self):
        df = pd.DataFrame(
            {
                "n_players": [5],
                "bonus_play_threshold": [4],
                "win_rate": [0.25],
            }
        )
        result = find_optimal_parameters(df)
        assert result == {"n_players": 5, "bonus_play_threshold": 4}

    def test_returns_first_when_tied(self):
        df = pd.DataFrame(
            {
                "n_players": [2, 3],
                "bonus_play_threshold": [1, 2],
                "win_rate": [0.5, 0.5],
            }
        )
        result = find_optimal_parameters(df)
        assert result["n_players"] == 2
        assert result["bonus_play_threshold"] == 1
