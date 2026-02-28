"""Visualize training curves from TensorBoard logs."""

import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def plot_training_metrics(log_dir="bld/rl_logs", output_dir="bld"):
    """Extract and plot training metrics from TensorBoard logs.

    Args:
        log_dir: Directory containing TensorBoard logs.
        output_dir: Directory to save plots.
    """
    log_path = Path(log_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    latest_run = max(log_path.iterdir(), key=lambda p: p.stat().st_mtime)
    event_file = list(latest_run.glob("events.out.tfevents.*"))[0]

    print(f"Reading logs from: {latest_run.name}")

    ea = EventAccumulator(str(event_file))
    ea.Reload()

    available_tags = ea.Tags()["scalars"]
    print(f"Available tags: {available_tags}")

    game_metrics = [
        ("game/win_rate", "Win Rate"),
        ("game/avg_cards_played", "Avg Cards Played"),
        ("rollout/ep_rew_mean", "Reward per Game"),
        ("game/avg_cards_per_turn", "Avg Cards per Turn"),
        ("game/avg_distance", "Avg Distance"),
    ]

    train_metrics = [
        ("train/explained_variance", "Explained Variance"),
        ("train/value_loss", "Value Loss"),
        ("train/policy_gradient_loss", "Policy Gradient Loss"),
        ("train/entropy_loss", "Entropy Loss"),
    ]

    game_available = [m for m in game_metrics if m[0] in available_tags]
    train_available = [m for m in train_metrics if m[0] in available_tags]

    if game_available:
        plot_metrics(
            ea, game_available, output_path / "game_metrics.png", "Game Metrics"
        )

    if train_available:
        plot_metrics(
            ea, train_available, output_path / "train_metrics.png", "Training Metrics"
        )

    print_summary(ea, available_tags)


def plot_metrics(ea, metrics, output_file, title):
    """Plot a set of metrics to a file.

    Args:
        ea: EventAccumulator instance.
        metrics: List of (tag, label) tuples.
        output_file: Path to save the plot.
        title: Title for the figure.
    """
    n_metrics = len(metrics)
    cols = min(2, n_metrics)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (tag, label) in enumerate(metrics):
        scalars = ea.Scalars(tag)
        steps = [s.step for s in scalars]
        values = [s.value for s in scalars]

        axes[idx].plot(steps, values, linewidth=1.5)
        axes[idx].set_title(label, fontsize=11)
        axes[idx].set_xlabel("Timesteps")
        axes[idx].set_ylabel(label)
        axes[idx].grid(True, alpha=0.3)

        if "win_rate" in tag.lower():
            axes[idx].set_ylim(0, max(0.1, max(values) * 1.1) if values else 0.1)
            axes[idx].axhline(
                y=0.014, color="r", linestyle="--", alpha=0.5, label="Baseline (1.4%)"
            )
            axes[idx].legend()

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")


def print_summary(ea, available_tags):
    """Print summary statistics for key metrics.

    Args:
        ea: EventAccumulator instance.
        available_tags: List of available scalar tags.
    """
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    key_metrics = [
        ("game/win_rate", "Final Win Rate"),
        ("game/avg_cards_played", "Final Avg Cards Played"),
        ("game/avg_cards_per_turn", "Final Avg Cards/Turn"),
        ("game/avg_distance", "Final Avg Distance"),
    ]

    for tag, label in key_metrics:
        if tag in available_tags:
            scalars = ea.Scalars(tag)
            if scalars:
                final_value = scalars[-1].value
                print(f"{label}: {final_value:.4f}")
        else:
            print(f"{label}: NOT LOGGED")


if __name__ == "__main__":
    plot_training_metrics()
