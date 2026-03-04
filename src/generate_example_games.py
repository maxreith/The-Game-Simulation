"""Generate example game logs showing every move for RL models."""

from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from game_env import TheGameEnv


def mask_fn(env):
    """Return valid action mask for MaskablePPO."""
    return env.unwrapped.action_masks()


def action_to_description(action: int, hand: list[int], stacks: list[int]) -> str:
    """Convert action index to human-readable description.

    Args:
        action: Action index (0-24 for 6 cards * 4 stacks + 1 end turn).
        hand: Current hand cards.
        stacks: Current stack top values [dec1, dec2, inc1, inc2].

    Returns:
        Human-readable action description.
    """
    if action == 24:
        return "End turn"

    card_idx = action // 4
    stack_idx = action % 4

    if card_idx >= len(hand):
        return f"Invalid: card_idx={card_idx} but hand has {len(hand)} cards"

    card = hand[card_idx]
    stack_top = stacks[stack_idx]
    stack_names = ["Dec1", "Dec2", "Inc1", "Inc2"]
    stack_name = stack_names[stack_idx]

    if stack_idx < 2:
        trick = card == stack_top + 10
    else:
        trick = card == stack_top - 10

    trick_str = " (TRICK PLAY!)" if trick else ""
    return f"Play {card} on {stack_name} (top={stack_top}){trick_str}"


def run_example_game(model, seed: int, n_players: int = 5, max_players: int = 3) -> str:
    """Run a single game and return formatted log.

    Args:
        model: Trained MaskablePPO model.
        seed: Random seed for reproducibility.
        n_players: Number of players.
        max_players: Max players for observation space (must match model).

    Returns:
        Formatted markdown string with game log.
    """
    env = TheGameEnv(n_players=n_players, max_players=max_players)
    env = ActionMasker(env, mask_fn)

    obs, info = env.reset(seed=seed)
    lines = [f"## Example Game (seed={seed})\n"]

    step_num = 0
    turn_num = 0
    cards_this_turn = 0
    inner_env = env.unwrapped
    current_player = inner_env.current_player_idx

    while True:
        step_num += 1

        hand = inner_env.hands[inner_env.current_player_idx].tolist()
        hand = [c for c in hand if c > 0]
        stacks = [
            inner_env.stacks[0].top,
            inner_env.stacks[1].top,
            inner_env.stacks[2].top,
            inner_env.stacks[3].top,
        ]
        deck_remaining = len(inner_env.remaining_deck)

        if cards_this_turn == 0:
            turn_num += 1
            lines.append(f"### Turn {turn_num}: Player {current_player + 1}")
            lines.append(f"- Hand: {sorted(hand)}")
            lines.append(
                f"- Stacks: Dec1={stacks[0]}, Dec2={stacks[1]}, "
                f"Inc1={stacks[2]}, Inc2={stacks[3]}"
            )
            lines.append(f"- Deck remaining: {deck_remaining} cards")
            lines.append("")

        action_masks = env.unwrapped.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        action = int(action)

        action_desc = action_to_description(action, hand, stacks)
        lines.append(f"**Step {step_num}:** {action_desc}")

        player_before = inner_env.current_player_idx
        obs, reward, terminated, truncated, info = env.step(action)

        if action == 24 or inner_env.current_player_idx != player_before:
            cards_this_turn = 0
            current_player = inner_env.current_player_idx
        else:
            cards_this_turn += 1

        if terminated or truncated:
            lines.append("")
            final_stacks = [
                inner_env.stacks[0].top,
                inner_env.stacks[1].top,
                inner_env.stacks[2].top,
                inner_env.stacks[3].top,
            ]
            if info.get("victory", False):
                lines.append("**VICTORY!** All cards played successfully.")
            else:
                lines.append(
                    f"**LOSS** - {info.get('total_cards_played', 0)} cards played"
                )
                remaining_hands = []
                for p in range(n_players):
                    h = inner_env.hands[p].tolist()
                    h = [c for c in h if c > 0]
                    if h:
                        remaining_hands.append(f"Player {p + 1}: {sorted(h)}")
                if remaining_hands:
                    lines.append("Remaining hands:")
                    for rh in remaining_hands:
                        lines.append(f"  - {rh}")
                lines.append(
                    f"Final stacks: Dec1={final_stacks[0]}, Dec2={final_stacks[1]}, "
                    f"Inc1={final_stacks[2]}, Inc2={final_stacks[3]}"
                )
            break

    lines.append("")
    return "\n".join(lines)


def generate_example_games(
    model_path: str,
    output_path: str,
    n_games: int = 3,
    n_players: int = 5,
    max_players: int = 3,
    base_seed: int = 42,
):
    """Generate example game logs for a trained model.

    Args:
        model_path: Path to trained MaskablePPO model.
        output_path: Path to output markdown file.
        n_games: Number of example games to generate.
        n_players: Number of players per game.
        max_players: Max players for observation space (must match model).
        base_seed: Base seed for reproducibility.
    """
    print(f"Loading model from {model_path}")
    model = MaskablePPO.load(model_path)

    lines = [
        f"# Example Games: {Path(model_path).stem}",
        "",
        f"Generated from model: `{Path(model_path).name}`",
        "",
        f"Configuration: {n_players} players, {n_games} games",
        "",
        "---",
        "",
    ]

    wins = 0
    for i in range(n_games):
        seed = base_seed + i
        print(f"Running game {i + 1}/{n_games} (seed={seed})...")
        game_log = run_example_game(
            model, seed=seed, n_players=n_players, max_players=max_players
        )
        lines.append(game_log)
        if "VICTORY" in game_log:
            wins += 1

    lines.insert(7, f"Results: {wins}/{n_games} wins ({100 * wins / n_games:.1f}%)")
    lines.insert(8, "")

    output = Path(output_path)
    output.write_text("\n".join(lines))
    print(f"Saved to {output_path}")


FINAL_MODELS = [
    ("Pure RL (sparse) @ 100M", "bld/sparse_100M_final.zip", 3),
    ("Pure RL (shaped) @ 100M", "bld/shaped_100M_final.zip", 5),
    ("BC+RL @ 100M", "bld/bc_rl_100M_final.zip", 5),
]


def main():
    """Generate example games for all 4 final models."""
    bld_dir = Path(__file__).parent.parent / "bld"

    for name, model_path, max_players in FINAL_MODELS:
        full_path = bld_dir.parent / model_path
        if not full_path.exists():
            print(f"[SKIP] {name}: model not found at {full_path}")
            continue

        output_name = model_path.split("/")[-1].replace(".zip", "_example_games.md")
        output_path = bld_dir / output_name

        print(f"\n{'=' * 60}")
        print(f"Generating example games for: {name}")
        print(f"{'=' * 60}")

        generate_example_games(
            model_path=str(full_path),
            output_path=str(output_path),
            n_games=3,
            n_players=5,
            max_players=max_players,
            base_seed=42,
        )


if __name__ == "__main__":
    main()
