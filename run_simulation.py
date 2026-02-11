"""Run simulations across multiple strategy variants and player counts."""

import itertools
from functools import partial
from pathlib import Path

import pandas as pd

from game_setup import run_simulation
from strategies import bonus_play_strategy, gemini_strategy


def build_strategy_variants():
    """Build a list of strategy variants with pre-configured parameters.

    Returns:
        A list of dicts with 'name' and 'func' keys. Each 'func' is a callable
        with uniform signature (player, stacks, remaining_deck) -> (player, stacks).
    """
    variants = []
    for threshold in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
        variants.append(
            {
                "name": f"bonus_play_t{threshold}",
                "func": partial(bonus_play_strategy, bonus_play_threshold=threshold),
            }
        )
    variants.append({"name": "gemini", "func": gemini_strategy})
    return variants


def build_param_grid(strategy_variants, n_players_list, n_games):
    """Generate all parameter combinations for simulation.

    Args:
        strategy_variants: List of strategy dicts from build_strategy_variants().
        n_players_list: List of player counts to test.
        n_games: Number of games per combination.

    Returns:
        A list of dicts with 'strategy_name', 'strategy_func', 'n_players', 'n_games'.
    """
    combinations = []
    for variant, n_players in itertools.product(strategy_variants, n_players_list):
        combinations.append(
            {
                "strategy_name": variant["name"],
                "strategy_func": variant["func"],
                "n_players": n_players,
                "n_games": n_games,
            }
        )
    return combinations


def run_all_simulations(param_combinations):
    """Run simulations for all parameter combinations.

    Args:
        param_combinations: List of param dicts from build_param_grid().

    Returns:
        A list of result dicts suitable for creating a DataFrame.
    """
    all_results = []
    for params in param_combinations:
        results = run_simulation(
            params["strategy_func"],
            n_games=params["n_games"],
            n_players=params["n_players"],
        )

        all_results.append(
            {
                "strategy": params["strategy_name"],
                "n_players": params["n_players"],
                "n_games": params["n_games"],
                "win_rate": results["win_rate"],
                "victories": len(results["victories"]),
                "losses": len(results["losses"]),
            }
        )

        print(
            f"✓ {params['strategy_name']}, "
            f"{params['n_players']} players: "
            f"{results['win_rate'] * 100:.1f}%"
        )

    return all_results


def main():
    """Run the full simulation grid and save results."""
    strategy_variants = build_strategy_variants()
    n_players_list = [2, 3, 4, 5, 6]
    n_games = 10000

    param_combinations = build_param_grid(strategy_variants, n_players_list, n_games)
    all_results = run_all_simulations(param_combinations)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(all_results)
    df.to_parquet(output_dir / "simulation_results.parquet", index=False)
    print(f"\nResults saved to {output_dir / 'simulation_results.parquet'}")
    print(df)


if __name__ == "__main__":
    main()
