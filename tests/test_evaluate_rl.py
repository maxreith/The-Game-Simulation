"""Tests for evaluate_rl.py functions."""

import json
import tempfile
from pathlib import Path

from sb3_contrib import MaskablePPO

from evaluate_rl import (
    evaluate_checkpoints,
    extract_step_count,
    save_results_json,
)
from train_rl import create_env


class TestExtractStepCount:
    def test_extracts_step_count_from_bc_rl_filename(self):
        assert extract_step_count("bc_rl_1000000_steps.zip") == 1_000_000

    def test_extracts_step_count_from_ppo_filename(self):
        assert extract_step_count("the_game_ppo_20000000_steps.zip") == 20_000_000

    def test_returns_none_for_invalid_format(self):
        assert extract_step_count("model.zip") is None
        assert extract_step_count("bc_rl_final.zip") is None

    def test_handles_various_step_counts(self):
        assert extract_step_count("model_500000_steps.zip") == 500_000
        assert extract_step_count("model_100_steps.zip") == 100


class TestSaveResultsJson:
    def test_saves_valid_json(self):
        results = [
            {
                "training_steps": 1_000_000,
                "win_rate": 0.05,
                "avg_cards_per_game": 87.0,
                "victories": 50,
                "losses": 950,
            }
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            save_results_json(results, output_path, "bc_rl", 5, 1000)

            with open(output_path) as f:
                data = json.load(f)

            assert data["model_type"] == "bc_rl"
            assert data["n_players"] == 5
            assert data["n_games"] == 1000
            assert "evaluated_at" in data
            assert len(data["checkpoints"]) == 1
            assert data["checkpoints"][0]["training_steps"] == 1_000_000
        finally:
            output_path.unlink()


class TestEvaluateCheckpoints:
    def test_returns_empty_list_for_missing_directory(self, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        result = evaluate_checkpoints(nonexistent, n_games=10, verbose=False)
        assert result == []

    def test_returns_empty_list_for_empty_directory(self, tmp_path):
        result = evaluate_checkpoints(tmp_path, n_games=10, verbose=False)
        assert result == []

    def test_evaluates_checkpoint_and_returns_results(self, tmp_path):
        env = create_env(n_players=5, n_envs=1)
        model = MaskablePPO("MlpPolicy", env, verbose=0, n_steps=64)
        model.learn(total_timesteps=64)

        checkpoint_path = tmp_path / "model_100_steps.zip"
        model.save(checkpoint_path)

        results = evaluate_checkpoints(
            tmp_path, n_games=5, n_players=5, seed=42, verbose=False
        )

        assert len(results) == 1
        assert results[0]["training_steps"] == 100
        assert "win_rate" in results[0]
        assert "avg_cards_per_game" in results[0]
        assert "victories" in results[0]
        assert "losses" in results[0]
        assert results[0]["victories"] + results[0]["losses"] == 5

    def test_sorts_results_by_step_count(self, tmp_path):
        env = create_env(n_players=5, n_envs=1)
        model = MaskablePPO("MlpPolicy", env, verbose=0, n_steps=64)
        model.learn(total_timesteps=64)

        (tmp_path / "model_3000000_steps.zip").write_bytes(
            (tmp_path / "temp.zip").write_bytes(b"") or b""
        )
        model.save(tmp_path / "model_1000000_steps.zip")
        model.save(tmp_path / "model_2000000_steps.zip")
        (tmp_path / "model_3000000_steps.zip").unlink()

        results = evaluate_checkpoints(
            tmp_path, n_games=2, n_players=5, seed=42, verbose=False
        )

        steps = [r["training_steps"] for r in results]
        assert steps == sorted(steps)
