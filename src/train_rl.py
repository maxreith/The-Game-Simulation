"""Train RL agent to play The Game using MaskablePPO."""

import os
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from game_env import TheGameEnv


def mask_fn(env):
    """Return valid action mask for MaskablePPO.

    Args:
        env: TheGameEnv instance.

    Returns:
        Boolean array indicating valid actions.
    """
    return env.action_masks()


def make_env(n_players=3):
    """Create a factory function for environment creation.

    Args:
        n_players: Number of players in the game.

    Returns:
        Factory function that creates a wrapped environment.
    """

    def _init():
        env = TheGameEnv(n_players=n_players)
        return ActionMasker(env, mask_fn)

    return _init


def create_env(n_players=3, n_envs=1, use_subproc=True):
    """Create vectorized environment for MaskablePPO training.

    Args:
        n_players: Number of players in the game.
        n_envs: Number of parallel environments.
        use_subproc: Use SubprocVecEnv (True) or DummyVecEnv (False).

    Returns:
        Vectorized environment with action masking.
    """
    env_fns = [make_env(n_players) for _ in range(n_envs)]

    if n_envs == 1:
        return DummyVecEnv(env_fns)
    elif use_subproc:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


def train(
    total_timesteps=500_000,
    n_players=3,
    n_envs=None,
    verbose=1,
    tensorboard_log=True,
):
    """Train MaskablePPO agent on The Game environment.

    Args:
        total_timesteps: Total training steps.
        n_players: Number of players in the game.
        n_envs: Number of parallel environments. Defaults to CPU count.
        verbose: Verbosity level for training output.
        tensorboard_log: Whether to enable tensorboard logging.

    Returns:
        Trained MaskablePPO model.
    """
    if n_envs is None:
        n_envs = os.cpu_count() or 1

    bld_dir = Path(__file__).parent.parent / "bld"
    bld_dir.mkdir(exist_ok=True)

    env = create_env(n_players=n_players, n_envs=n_envs)

    if verbose:
        print(f"Training with {n_envs} parallel environments")

    log_path = str(bld_dir / "rl_logs") if tensorboard_log else None

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        tensorboard_log=log_path,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(bld_dir / "the_game_ppo")

    return model


def main():
    """Entry point for training script."""
    train()


if __name__ == "__main__":
    main()
