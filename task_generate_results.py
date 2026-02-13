from pathlib import Path
import subprocess

SRC = Path("src")
BLD = Path("bld")


def task_run_strategy_simulation(
    code: Path = SRC / "simulate_strategies.py",
    produces: Path = BLD / "simulation_results.parquet",
):
    """Runs the strategy simulations and saves results as a parquet file."""
    subprocess.run(["python", code], check=True)


def task_evaluate_shuffle_quality(
    code: Path = SRC / "simulate_shuffle_quality.py",
    depends: Path = BLD / "simulation_results.parquet",
    produces: Path = BLD / "shuffle_quality_results.parquet",
):
    """Evaluates shuffle quality using the optimal parameters from strategy simulations."""
    subprocess.run(["python", code], check=True)


def task_generate_plots(
    code: Path = SRC / "generate_plots.py",
    depends: list = [
        BLD / "simulation_results.parquet",
        BLD / "shuffle_quality_results.parquet",
    ],
    produces: list = [BLD / "strategy_evaluation.png", BLD / "shuffle_evaluation.png"],
):
    """Generates visualization plots from simulation results."""
    subprocess.run(["pixi", "run", "python", code], check=True)
