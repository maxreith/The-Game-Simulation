"""Evaluate trained RL agent against baseline strategies."""

from functools import partial
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from game_env import TheGameEnv
from game_setup import run_simulation
from strategies import bonus_play_strategy
from train_bc_rl import BCPolicyNetwork


def evaluate_rl_agent(model, n_games=1000, n_players=5, seed=None):
    """Run games with RL agent and compute win rate and extended statistics.

    Args:
        model: Trained MaskablePPO model.
        n_games: Number of games to play.
        n_players: Number of players.
        seed: Random seed for reproducibility.

    Returns:
        Dict with win_rate, victories, losses, cards_per_game, cards_per_turn,
        distances, trick_plays_per_game.
    """
    env = TheGameEnv(n_players=n_players)
    victories = 0
    cards_per_game = []
    cards_per_turn_all = []
    distances_all = []
    trick_plays_per_game = []

    for game_idx in range(n_games):
        game_seed = seed + game_idx if seed is not None else None
        obs, info = env.reset(seed=game_seed)
        terminated = False

        game_distances = []
        game_trick_plays = 0
        cards_this_turn = 0

        while not terminated:
            action_mask = env.action_masks()

            # Snapshot piles before step to compute distance & trick plays
            state = env.game.state if hasattr(env, "game") else None
            piles_before = [p.top for p in state.piles] if state is not None else None

            action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
            obs, reward, terminated, truncated, info = env.step(action)

            # Detect end-of-turn action (pass) vs card play
            # action 0 is typically "pass/end turn" in TheGameEnv
            is_pass = action == 0

            if not is_pass:
                cards_this_turn += 1

                # Compute distance if we have pile info
                if piles_before is not None and state is not None:
                    piles_after = [p.top for p in state.piles]
                    for before, after in zip(piles_before, piles_after):
                        if before != after:
                            dist = abs(after - before)
                            game_distances.append(dist)
                            # Trick play: moving backward by exactly 10
                            # Ascending pile trick: after == before - 10
                            # Descending pile trick: after == before + 10
                            if dist == 10:
                                game_trick_plays += 1
            else:
                if cards_this_turn > 0:
                    cards_per_turn_all.append(cards_this_turn)
                cards_this_turn = 0

        # Flush last turn if game ended mid-turn
        if cards_this_turn > 0:
            cards_per_turn_all.append(cards_this_turn)

        if info.get("victory", False):
            victories += 1
        cards_per_game.append(env.total_cards_played)
        distances_all.extend(game_distances)
        trick_plays_per_game.append(game_trick_plays)

    return {
        "win_rate": victories / n_games,
        "victories": victories,
        "losses": n_games - victories,
        "cards_per_game": cards_per_game,
        "cards_per_turn": cards_per_turn_all,
        "distances": distances_all,
        "trick_plays_per_game": trick_plays_per_game,
    }


def replay_single_game(model, n_players=5, seed=None):
    """Replay a single game with the RL agent.

    Args:
        model: Trained MaskablePPO model.
        n_players: Number of players.
        seed: Random seed for reproducibility.

    Returns:
        True if game was won, False otherwise.
    """
    env = TheGameEnv(n_players=n_players)
    obs, info = env.reset(seed=seed)
    terminated = False

    while not terminated:
        action_mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
        obs, reward, terminated, truncated, info = env.step(action)

    return info.get("victory", False)


def evaluate_baseline(n_games=1000, n_players=5, bonus_threshold=2, seed=None):
    """Run games with bonus_play_strategy and compute win rate.

    Args:
        n_games: Number of games to play.
        n_players: Number of players.
        bonus_threshold: Threshold parameter for bonus_play_strategy.
        seed: Random seed for reproducibility.

    Returns:
        Dict with win_rate, victories, losses.
    """
    if seed is not None:
        np.random.seed(seed)

    strategy = partial(bonus_play_strategy, bonus_play_threshold=bonus_threshold)
    results = run_simulation(strategy, n_games=n_games, n_players=n_players)

    return {
        "win_rate": results["win_rate"],
        "victories": len(results["victories"]),
        "losses": len(results["losses"]),
    }


def _stats(arr):
    """Return (mean, median) for a list of numbers."""
    if not arr:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.median(arr))


def evaluate_bc_only(bc_model_path, n_games=1000, n_players=5, seed=None):
    """Evaluate BC-only model (before RL fine-tuning).

    Args:
        bc_model_path: Path to saved BC policy weights (.pt file).
        n_games: Number of games to play.
        n_players: Number of players.
        seed: Random seed for reproducibility.

    Returns:
        Dict with win_rate, avg_cards.
    """
    env = TheGameEnv(n_players=n_players)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    bc_model = BCPolicyNetwork(obs_dim, action_dim)
    bc_model.load_state_dict(torch.load(bc_model_path, weights_only=True))
    bc_model.eval()

    victories = 0
    cards_per_game = []

    for game_idx in range(n_games):
        game_seed = seed + game_idx if seed is not None else None
        obs, _ = env.reset(seed=game_seed)
        terminated = False

        while not terminated:
            mask = env.action_masks()
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
                logits = bc_model(obs_tensor, mask_tensor)
                action = logits.argmax(dim=1).item()
            obs, _, terminated, _, info = env.step(action)

        cards_per_game.append(env.total_cards_played)
        if info.get("victory", False):
            victories += 1

    return {
        "win_rate": victories / n_games,
        "avg_cards": float(np.mean(cards_per_game)),
    }


def main():
    """Evaluate 4 models and compare against baseline."""
    bld_dir = Path(__file__).parent.parent / "bld"

    models = [
        ("Pure RL (sparse) @ 100M", bld_dir / "sparse_100M_final.zip"),
        ("Pure RL (shaped) @ 100M", bld_dir / "shaped_100M_final.zip"),
        ("BC+RL @ 100M", bld_dir / "bc_rl_checkpoints" / "bc_rl_100000000_steps.zip"),
        ("BC+RL @ 500M", bld_dir / "bc_rl_500M_final.zip"),
    ]

    n_games = 10000
    n_players = 5
    seed = 42

    output_lines = []
    output_lines.append(
        f"Evaluating models ({n_games} games, {n_players} players, seed={seed})\n"
    )
    output_lines.append("=" * 60)

    results = []

    for name, path in models:
        if not path.exists():
            output_lines.append(f"[SKIP] {name}: checkpoint not found at {path}")
            results.append((name, None, None))
            continue

        output_lines.append(f"Evaluating {name}...")
        model = MaskablePPO.load(path)
        eval_result = evaluate_rl_agent(model, n_games, n_players, seed)
        win_rate = eval_result["win_rate"]
        avg_cards = float(np.mean(eval_result["cards_per_game"]))
        results.append((name, win_rate, avg_cards))
        output_lines.append(f"  Win rate: {win_rate:.1%}, Avg cards: {avg_cards:.1f}")

    bc_path = bld_dir / "bc_policy.pt"
    if bc_path.exists():
        output_lines.append("\nEvaluating BC-only (before RL fine-tuning)...")
        bc_result = evaluate_bc_only(bc_path, n_games, n_players, seed)
        results.append(("BC-only", bc_result["win_rate"], bc_result["avg_cards"]))
        output_lines.append(
            f"  Win rate: {bc_result['win_rate']:.1%}, Avg cards: {bc_result['avg_cards']:.1f}"
        )
    else:
        output_lines.append(f"\n[SKIP] BC-only: model not found at {bc_path}")
        results.append(("BC-only", None, None))

    output_lines.append("\nEvaluating baseline (bonus_play_strategy)...")
    baseline_result = evaluate_baseline(
        n_games, n_players, bonus_threshold=2, seed=seed
    )
    baseline_win_rate = baseline_result["win_rate"]
    results.append(("Baseline (bonus_play)", baseline_win_rate, None))
    output_lines.append(f"  Win rate: {baseline_win_rate:.1%}")

    output_lines.append("\n" + "=" * 60)
    output_lines.append("COMPARISON TABLE")
    output_lines.append("=" * 60)
    output_lines.append(f"{'Model':<30} {'Win Rate':>10} {'Avg Cards':>12}")
    output_lines.append("-" * 54)
    for name, win_rate, avg_cards in results:
        if win_rate is None:
            output_lines.append(f"{name:<30} {'N/A':>10} {'N/A':>12}")
        elif avg_cards is None:
            output_lines.append(f"{name:<30} {win_rate:>9.1%} {'N/A':>12}")
        else:
            output_lines.append(f"{name:<30} {win_rate:>9.1%} {avg_cards:>11.1f}")
    output_lines.append("=" * 60)

    output_text = "\n".join(output_lines)

    output_file = bld_dir / "evaluation_results.txt"
    bld_dir.mkdir(parents=True, exist_ok=True)
    output_file.write_text(output_text)


if __name__ == "__main__":
    main()
