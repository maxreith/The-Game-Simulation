"""Simulate games with varying Gemini thinking levels."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import pandas as pd

from game_setup import run_game
from strategies import gemini_strategy

pd.options.future.infer_string = True

THINKING_LEVELS = ["minimal", "low", "medium"]
N_GAMES_PER_LEVEL = 1


def _run_single_game(level: str, game_num: int) -> dict:
    """Run a single game with the given thinking level.

    Args:
        level: Gemini thinking level.
        game_num: Game number (1-indexed).

    Returns:
        Dict with game results.
    """
    strategy = partial(gemini_strategy, thinking_level=level)
    result = run_game(strategy)
    return {
        "thinking_level": level,
        "game_number": game_num,
        "turns": result["turns"],
        "victory": result["victory"],
        "cards_remaining": result["cards_remaining"],
    }


def run_thinking_level_simulation(
    thinking_levels: list[str] = THINKING_LEVELS,
    n_games_per_level: int = N_GAMES_PER_LEVEL,
    parallel: bool = True,
) -> pd.DataFrame:
    """Run games for each thinking level and collect results.

    Args:
        thinking_levels: List of thinking levels to test.
        n_games_per_level: Number of games to run per thinking level.
        parallel: Whether to run games in parallel.

    Returns:
        DataFrame with columns: thinking_level, game_number, turns, victory, cards_remaining.
    """
    tasks = [
        (level, game_num)
        for level in thinking_levels
        for game_num in range(1, n_games_per_level + 1)
    ]

    if parallel:
        results = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_run_single_game, level, game_num): (level, game_num)
                for level, game_num in tasks
            }
            for future in as_completed(futures):
                level, game_num = futures[future]
                result = future.result()
                results.append(result)
                print(
                    f"Completed game {game_num}/{n_games_per_level} "
                    f"with thinking_level={level} (turns={result['turns']})"
                )
    else:
        results = []
        for level, game_num in tasks:
            print(
                f"Running game {game_num}/{n_games_per_level} with thinking_level={level}"
            )
            result = _run_single_game(level, game_num)
            results.append(result)

    df = pd.DataFrame(results)
    return df


def main() -> None:
    """Run simulation and save results."""
    start_time = time.time()

    df = run_thinking_level_simulation()

    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60

    df.to_parquet("bld/gemini_thinking_results.parquet")
    print("\nSaved results to bld/gemini_thinking_results.parquet")
    print(
        f"Total simulation time: {elapsed_minutes:.1f} minutes ({elapsed_time:.1f} seconds)"
    )
    print(df)


if __name__ == "__main__":
    main()
