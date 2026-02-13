"""
Simulation script to:
1. Find optimal parameters for bonus_play_strategy
2. Test win rates across different shuffle qualities
"""

import pandas as pd
from functools import partial
from game_setup import run_simulation
from strategies import bonus_play_strategy


def find_optimal_parameters(
    simulation_results_path: str = "bld/simulation_results.parquet",
):
    """
    Read simulation results and find the combination with highest win rate.

    Args:
        simulation_results_path: Path to the parquet file with simulation results

    Returns:
        dict: Best parameters and their win rate
    """
    # Read simulation results
    df = pd.read_parquet(simulation_results_path)

    # Find best parameters
    best_idx = df["win_rate"].idxmax()
    best_params = df.loc[best_idx]

    return {
        "n_players": int(best_params["n_players"]),
        "bonus_play_threshold": int(best_params["bonus_play_threshold"]),
        "win_rate": best_params["win_rate"],
        "all_results": df,
    }


def evaluate_shuffle_qualities(optimal_params: dict, n_games: int = 100):
    """
    Test win rates across different shuffle qualities using optimal parameters.

    Args:
        optimal_params: Dictionary with 'n_players' and 'bonus_play_threshold'
        n_games: Number of games to simulate per shuffle quality

    Returns:
        pandas.DataFrame: Results for each shuffle quality
    """
    # Test different shuffle qualities
    shuffle_qualities = [1, 2, 5, 10, 20, 50, 100, 200]
    results = []

    for n_shuffles in shuffle_qualities:
        configured_strategy = partial(
            bonus_play_strategy,
            bonus_play_threshold=optimal_params["bonus_play_threshold"],
        )
        result = run_simulation(
            strategy=configured_strategy,
            n_games=n_games,
            n_players=optimal_params["n_players"],
            n_shuffles=n_shuffles,
            use_custom_shuffle=True,
        )

        results.append(
            {
                "n_shuffles": n_shuffles,
                "shuffle_quality": get_shuffle_description(n_shuffles),
                "victories": len(result["victories"]),
                "losses": len(result["losses"]),
                "win_rate": result["win_rate"],
            }
        )

    # Convert to DataFrame
    df = pd.DataFrame(results)

    return df


def get_shuffle_description(n_shuffles: int) -> str:
    """Return a human-readable description of shuffle quality."""
    if n_shuffles <= 2:
        return "Very Poor"
    elif n_shuffles <= 10:
        return "Poor"
    elif n_shuffles <= 50:
        return "Moderate"
    elif n_shuffles <= 100:
        return "Good"
    else:
        return "Excellent"


def main():
    """Run the complete simulation pipeline."""
    # Phase 1: Find optimal parameters from simulation results
    optimal_params = find_optimal_parameters()

    # Phase 2: Test shuffle qualities (using even more games for accurate comparison)
    n_games_shuffle_test = 1000
    shuffle_results = evaluate_shuffle_qualities(
        optimal_params, n_games=n_games_shuffle_test
    )

    # Save results
    shuffle_results.to_parquet("bld/shuffle_quality_results.parquet", index=False)


if __name__ == "__main__":
    main()
