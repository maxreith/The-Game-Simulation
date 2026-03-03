"""Generate expert demonstrations for behavioral cloning.

Collects (observation, action, action_mask) tuples by running TheGameEnv
with expert actions computed from bonus_play_strategy logic.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from game_env import TheGameEnv
from utils import GameOverError, identify_min_distance_card


def get_expert_action(env: TheGameEnv, bonus_play_threshold: int = 4) -> int:
    """Compute expert action for current state using bonus_play_strategy logic.

    Args:
        env: TheGameEnv instance in current state.
        bonus_play_threshold: Maximum distance for bonus plays beyond minimum.

    Returns:
        Action index to take (card_idx * 4 + stack_idx, or end_turn_action).
    """
    hand = env.hands[env.current_player_idx]
    stacks = env.stacks
    min_required = env._min_cards_required()
    end_turn_action = env.hand_size * 4

    if len(hand) == 0:
        return end_turn_action

    try:
        best_card, best_stack, min_diff = identify_min_distance_card(hand, stacks)
    except GameOverError:
        return end_turn_action

    if env.cards_played_this_turn >= min_required and min_diff > bonus_play_threshold:
        return end_turn_action

    card_idx = np.where(hand == best_card)[0][0]
    return card_idx * 4 + best_stack


def generate_expert_demonstrations(
    n_games: int = 10000,
    n_players: int = 5,
    bonus_play_threshold: int = 4,
    output_path: Path | str | None = None,
    seed: int | None = None,
    verbose: bool = True,
):
    """Collect expert demonstrations by running games with bonus_play_strategy.

    Args:
        n_games: Number of games to collect demonstrations from.
        n_players: Number of players per game.
        bonus_play_threshold: Maximum distance for bonus plays.
        output_path: Path to save the demonstrations. If None, returns in memory.
        seed: Random seed for reproducibility.
        verbose: Whether to show progress bar.

    Returns:
        Tuple of (observations, actions, action_masks) arrays if output_path is None.
        Otherwise saves to file and returns the output path.
    """
    env = TheGameEnv(
        n_players=n_players,
        reward_per_card=0.02,
        win_reward=10.0,
        loss_penalty=0.0,
        trick_play_reward=1.0,
        distance_penalty_scale=0.003,
        progress_reward_scale=3.0,
        stack_health_scale=0.01,
        phase_multiplier_scale=0.5,
    )

    observations = []
    actions = []
    action_masks = []
    wins = 0
    total_cards = []

    game_range = (
        tqdm(range(n_games), desc="Collecting demos") if verbose else range(n_games)
    )

    for game_idx in game_range:
        game_seed = seed + game_idx if seed is not None else None
        obs, _ = env.reset(seed=game_seed)
        terminated = False

        while not terminated:
            mask = env.action_masks()
            expert_action = get_expert_action(env, bonus_play_threshold)

            if not mask[expert_action]:
                valid_actions = np.where(mask)[0]
                if len(valid_actions) == 0:
                    break
                expert_action = valid_actions[0]

            observations.append(obs.copy())
            actions.append(expert_action)
            action_masks.append(mask.copy())

            obs, _, terminated, _, info = env.step(expert_action)

        total_cards.append(info.get("total_cards_played", 0))
        if info.get("victory", False):
            wins += 1

    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    action_masks = np.array(action_masks, dtype=bool)

    if verbose:
        print(f"Collected {len(observations)} state-action pairs from {n_games} games")
        print(f"Win rate: {wins / n_games:.2%}")
        print(f"Avg cards played: {np.mean(total_cards):.1f}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            observations=observations,
            actions=actions,
            action_masks=action_masks,
        )
        if verbose:
            print(f"Saved to {output_path}")
        return output_path

    return observations, actions, action_masks


def load_expert_demonstrations(path: Path | str):
    """Load expert demonstrations from file.

    Args:
        path: Path to the saved demonstrations.

    Returns:
        Tuple of (observations, actions, action_masks) arrays.
    """
    data = np.load(path)
    return data["observations"], data["actions"], data["action_masks"]


def main():
    """Generate and save expert demonstrations."""
    bld_dir = Path(__file__).parent.parent / "bld"
    output_path = bld_dir / "expert_demonstrations.npz"

    generate_expert_demonstrations(
        n_games=10000,
        n_players=5,
        bonus_play_threshold=4,
        output_path=output_path,
        seed=42,
        verbose=True,
    )


if __name__ == "__main__":
    main()
