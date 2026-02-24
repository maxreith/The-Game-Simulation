"""Evaluate trained RL agent against baseline strategies."""

from functools import partial
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from game_env import TheGameEnv
from game_setup import run_simulation
from strategies import bonus_play_strategy


def evaluate_rl_agent(model, n_games=1000, n_players=3, seed=None):
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


def replay_single_game(model, n_players=3, seed=None, verbose=True):
    """Replay a single game with the RL agent and print every step."""
    env = TheGameEnv(n_players=n_players)
    obs, info = env.reset(seed=seed)
    terminated = False
    step = 0
    turn = 0

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"SAMPLE GAME  (seed={seed}, players={n_players})")
        print(f"{'=' * 60}")

    cards_this_turn = 0

    while not terminated:
        action_mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)

        # Capture state BEFORE the step
        game = getattr(env, "game", None)
        state = getattr(game, "state", None) if game else None
        pile_tops_before = None
        hand_before = None
        current_player = "?"
        if state is not None:
            pile_tops_before = [p.top for p in state.piles]
            current_player = getattr(state, "current_player", "?")
            if hasattr(state, "hands"):
                hand_before = sorted(state.hands[state.current_player])

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        is_pass = action == 0
        if verbose:
            if is_pass:
                print(f"  → END TURN (played {cards_this_turn} card(s))\n")
                turn += 1
                cards_this_turn = 0
            else:
                # Decode action to (card, stack)
                decoded = None
                if hasattr(env, "decode_action"):
                    decoded = env.decode_action(action)
                elif hasattr(env, "_decode_action"):
                    decoded = env._decode_action(action)
                elif hasattr(env, "action_to_card_stack"):
                    decoded = env.action_to_card_stack.get(action)

                stack_names = {
                    0: "declining stack 1",
                    1: "declining stack 2",
                    2: "inclining stack 1",
                    3: "inclining stack 2",
                }
                hand_str = ",".join(str(c) for c in hand_before) if hand_before else "?"

                if decoded is not None:
                    card, stack_idx = decoded
                    stack_name = stack_names.get(stack_idx, f"stack {stack_idx}")
                    top_before = (
                        pile_tops_before[stack_idx] if pile_tops_before else "?"
                    )
                    print(
                        f"  Player {current_player}: play card {card} on {stack_name} "
                        f"with current top {top_before}. "
                        f"Hand: [{hand_str}]  (reward={reward:.3f})"
                    )
                else:
                    print(
                        f"  Player {current_player}: action {action}. "
                        f"Hand: [{hand_str}]  (reward={reward:.3f})"
                    )
                cards_this_turn += 1

    if verbose:
        victory = info.get("victory", False)
        print(f"\n{'=' * 60}")
        print(f"GAME OVER — {'VICTORY 🎉' if victory else 'DEFEAT 💀'}")
        print(f"Total cards played: {env.total_cards_played}")
        print(f"{'=' * 60}\n")

    return info.get("victory", False)


def evaluate_baseline(n_games=1000, n_players=3, bonus_threshold=2, seed=None):
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


def main():
    """Compare RL agent against baseline strategy."""
    model_path = Path(__file__).parent.parent / "bld" / "the_game_ppo.zip"

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Run train_rl.py first to train the model.")
        return

    print("Loading trained RL model...")
    model = MaskablePPO.load(model_path)

    n_games = 1000
    n_players = 3
    seed = 42

    print(f"\nEvaluating on {n_games} games with {n_players} players...\n")

    print("Running RL agent...")
    rl_results = evaluate_rl_agent(model, n_games, n_players, seed)

    avg_cpg, med_cpg = _stats(rl_results["cards_per_game"])
    avg_cpt, med_cpt = _stats(rl_results["cards_per_turn"])
    avg_dist, med_dist = _stats(rl_results["distances"])
    avg_tp, med_tp = _stats(rl_results["trick_plays_per_game"])

    print(f"  Win rate:                    {rl_results['win_rate'] * 100:.1f}%")
    print(f"  Cards played / game  — avg: {avg_cpg:.1f},  median: {med_cpg:.1f}")
    print(f"  Cards played / turn  — avg: {avg_cpt:.1f},  median: {med_cpt:.1f}")
    print(f"  Distance per card    — avg: {avg_dist:.1f},  median: {med_dist:.1f}")
    print(f"  Trick plays / game   — avg: {avg_tp:.1f},  median: {med_tp:.1f}")

    print("\nRunning bonus_play_strategy (threshold=2)...")
    baseline_results = evaluate_baseline(
        n_games, n_players, bonus_threshold=2, seed=seed
    )
    print(f"  Win rate: {baseline_results['win_rate'] * 100:.1f}%")

    print("\n" + "=" * 40)
    print("COMPARISON")
    print("=" * 40)
    diff = (rl_results["win_rate"] - baseline_results["win_rate"]) * 100
    print(f"RL Agent:     {rl_results['win_rate'] * 100:5.1f}%")
    print(f"Baseline:     {baseline_results['win_rate'] * 100:5.1f}%")
    print(f"Difference:   {diff:+5.1f}%")

    if diff > 0:
        print("\nRL agent outperforms baseline!")
    elif diff < 0:
        print("\nBaseline outperforms RL agent.")
    else:
        print("\nPerformance is equal.")

    # Print a randomly sampled game in full detail
    rng = np.random.default_rng(seed)
    sample_seed = int(rng.integers(0, 10_000))
    print(f"\nSampling a single game (seed={sample_seed}) for detailed inspection...")
    replay_single_game(model, n_players=n_players, seed=sample_seed, verbose=True)


if __name__ == "__main__":
    main()
