"""Generate visualization plots from simulation results."""

import pandas as pd
import matplotlib.pyplot as plt

pd.options.future.infer_string = True


def plot_strategy_evaluation(df: pd.DataFrame, output_path: str) -> None:
    """Create line plot of win rate by number of players for each threshold.

    Args:
        df: DataFrame with columns n_players, bonus_play_threshold, win_rate.
        output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    thresholds = sorted(df["bonus_play_threshold"].unique())
    for threshold in thresholds:
        subset = df[df["bonus_play_threshold"] == threshold].sort_values("n_players")
        ax.plot(
            subset["n_players"],
            subset["win_rate"],
            marker="o",
            label=f"Threshold {threshold}",
        )

    ax.set_xlabel("Number of Players")
    ax.set_ylabel("Win Rate")
    ax.set_title("Strategy Evaluation: Win Rate by Number of Players")
    ax.legend(title="Bonus Play Threshold", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_shuffle_evaluation(
    df: pd.DataFrame, optimal_params: dict, output_path: str
) -> None:
    """Create line plot of win rate by number of shuffles.

    Args:
        df: DataFrame with columns n_shuffles, win_rate.
        optimal_params: Dict with n_players and bonus_play_threshold used.
        output_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df_sorted = df.sort_values("n_shuffles")
    ax.plot(df_sorted["n_shuffles"], df_sorted["win_rate"], marker="o", linewidth=2)

    ax.set_xlabel("Number of Shuffles")
    ax.set_ylabel("Win Rate")
    ax.set_title(
        f"Shuffle Quality Evaluation\n"
        f"(n_players={optimal_params['n_players']}, "
        f"bonus_play_threshold={optimal_params['bonus_play_threshold']})"
    )
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def find_optimal_parameters(df: pd.DataFrame) -> dict:
    """Find the parameters with the highest win rate.

    Args:
        df: DataFrame with columns n_players, bonus_play_threshold, win_rate.

    Returns:
        Dict with n_players and bonus_play_threshold for the best result.
    """
    best_idx = df["win_rate"].idxmax()
    return df.loc[best_idx, ["n_players", "bonus_play_threshold"]].to_dict()


def main() -> None:
    """Load data and generate plots."""
    sim_results = pd.read_parquet("bld/simulation_results.parquet")
    shuffle_results = pd.read_parquet("bld/shuffle_quality_results.parquet")

    optimal_params = find_optimal_parameters(sim_results)

    plot_strategy_evaluation(sim_results, "bld/strategy_evaluation.png")
    plot_shuffle_evaluation(
        shuffle_results, optimal_params, "bld/shuffle_evaluation.png"
    )


if __name__ == "__main__":
    main()
