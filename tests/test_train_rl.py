"""Tests for RL training script."""

import numpy as np

from train_rl import create_env, train


class TestCreateEnv:
    """Tests for environment creation."""

    def test_creates_wrapped_env(self):
        """Environment is a vectorized env with action masking support."""
        env = create_env(n_players=3, n_envs=1)
        assert hasattr(env, "env_method")
        env.reset()
        masks = env.env_method("action_masks")
        assert len(masks) == 1
        assert masks[0].shape == (25,)  # hand_size=6, stacks=4, + end turn
        env.close()

    def test_env_reset_works(self):
        """Vectorized environment resets successfully."""
        env = create_env(n_players=3, n_envs=1)
        obs = env.reset()
        assert obs is not None
        assert obs.shape[0] == 1  # n_envs=1
        env.close()


class TestTrain:
    """Tests for training function."""

    def test_short_training_runs(self):
        """Short training completes without errors."""
        model = train(
            total_timesteps=100,
            n_players=3,
            n_envs=1,
            verbose=0,
            tensorboard_log=False,
        )
        assert model is not None

    def test_model_can_predict(self):
        """Trained model can make predictions."""
        model = train(
            total_timesteps=100,
            n_players=3,
            n_envs=1,
            verbose=0,
            tensorboard_log=False,
        )
        env = create_env(n_players=3, n_envs=1)
        obs = env.reset()

        masks = env.env_method("action_masks")
        action, _ = model.predict(obs, action_masks=np.array(masks))
        assert 0 <= action < 25  # hand_size=6, stacks=4, + end turn
        env.close()
