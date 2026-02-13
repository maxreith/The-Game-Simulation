"""
Generate simulation_results.parquet by testing all parameter combinations.
This should be run once to create the data that find_optimal_parameters reads.
"""

import pandas as pd
from functools import partial
from itertools import product
from game_setup import run_simulation
from strategies import bonus_play_strategy


def generate_simulation_results(
    n_games: int = 1000, output_path: str = "simulation_results.parquet"
):
    """
    Test different parameter combinations and save results to parquet.

    Args:
        n_games: Number of games to simulate per parameter combination
        output_path: Path to save the parquet file
    """
    # Define parameter grid
    n_players_options = [2, 3, 4, 5, 6]
    bonus_play_threshold_options = [1, 2, 3, 4, 5, 6, 7, 8]

    results = []

    # Test all combinations
    for n_players, threshold in product(
        n_players_options, bonus_play_threshold_options
    ):
        configured_strategy = partial(
            bonus_play_strategy, bonus_play_threshold=threshold
        )
        result = run_simulation(
            strategy=configured_strategy,
            n_games=n_games,
            n_players=n_players,
        )

        results.append(
            {
                "n_players": n_players,
                "bonus_play_threshold": threshold,
                "win_rate": result["win_rate"],
                "victories": len(result["victories"]),
                "losses": len(result["losses"]),
            }
        )

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    generate_simulation_results()
