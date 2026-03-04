"""Train RL agent using Behavioral Cloning pre-training followed by PPO fine-tuning.

Two-phase approach:
1. Pre-train neural network to imitate bonus_play_strategy via supervised learning
2. Initialize MaskablePPO with BC weights and fine-tune with RL to 100M steps

Outputs:
- bld/bc_rl_checkpoints/bc_rl_*_steps.zip (checkpoints every 10M steps)
- bld/bc_rl_100M_final.zip (final model)
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from tqdm import tqdm

from game_env import TheGameEnv
from generate_expert_data import (
    generate_expert_demonstrations,
    load_expert_demonstrations,
)
from train_rl import (
    GameMetricsCallback,
    SPARSE_REWARDS,
    create_env,
)


class BCPolicyNetwork(nn.Module):
    """Policy network matching SB3's MlpPolicy with net_arch=[256, 256].

    Architecture must match SB3's policy network exactly for weight transfer.
    Uses Sequential indices 0, 2 for Linear layers (1, 3 are activations).
    """

    def __init__(self, obs_dim: int, action_dim: int):
        """Initialize BC policy network.

        Args:
            obs_dim: Dimension of observation space.
            action_dim: Dimension of action space.
        """
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.action_net = nn.Linear(256, action_dim)

    def forward(self, obs, action_mask=None):
        """Forward pass through policy network.

        Args:
            obs: Observation tensor of shape (batch, obs_dim).
            action_mask: Optional boolean mask of shape (batch, action_dim).

        Returns:
            Logits tensor of shape (batch, action_dim).
        """
        features = self.policy_net(obs)
        logits = self.action_net(features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)
        return logits


def train_behavioral_cloning(
    observations: np.ndarray,
    actions: np.ndarray,
    action_masks: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    validation_split: float = 0.1,
    patience: int = 10,
    device: str | None = None,
    verbose: bool = True,
):
    """Train BC policy network via supervised learning.

    Args:
        observations: Expert observations of shape (n_samples, obs_dim).
        actions: Expert actions of shape (n_samples,).
        action_masks: Action masks of shape (n_samples, action_dim).
        epochs: Maximum training epochs.
        batch_size: Mini-batch size for training.
        learning_rate: Learning rate for optimizer.
        validation_split: Fraction of data for validation.
        patience: Early stopping patience (epochs without improvement).
        device: Torch device ("cuda" or "cpu"). Auto-detected if None.
        verbose: Whether to print training progress.

    Returns:
        Trained BCPolicyNetwork on CPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_samples = len(observations)
    n_val = int(n_samples * validation_split)
    indices = np.random.permutation(n_samples)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    obs_dim = observations.shape[1]
    action_dim = action_masks.shape[1]
    model = BCPolicyNetwork(obs_dim, action_dim).to(device)

    train_obs = torch.tensor(
        observations[train_idx], dtype=torch.float32, device=device
    )
    train_actions = torch.tensor(actions[train_idx], dtype=torch.long, device=device)
    train_masks = torch.tensor(action_masks[train_idx], dtype=torch.bool, device=device)

    val_obs = torch.tensor(observations[val_idx], dtype=torch.float32, device=device)
    val_actions = torch.tensor(actions[val_idx], dtype=torch.long, device=device)
    val_masks = torch.tensor(action_masks[val_idx], dtype=torch.bool, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    n_train = len(train_idx)

    epoch_range = tqdm(range(epochs), desc="BC Training") if verbose else range(epochs)

    for epoch in epoch_range:
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0

        for i in range(0, n_train, batch_size):
            batch_idx = perm[i : i + batch_size]
            batch_obs = train_obs[batch_idx]
            batch_actions = train_actions[batch_idx]
            batch_masks = train_masks[batch_idx]

            optimizer.zero_grad()
            logits = model(batch_obs, batch_masks)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)

        train_loss = epoch_loss / n_train

        model.eval()
        with torch.no_grad():
            val_logits = model(val_obs, val_masks)
            val_loss = criterion(val_logits, val_actions).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_actions).float().mean().item()

        if verbose:
            epoch_range.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.2%}",
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model = model.cpu()

    if verbose:
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation accuracy: {val_acc:.2%}")

    return model


def load_bc_weights_into_ppo(bc_model: BCPolicyNetwork, ppo: MaskablePPO):
    """Transfer BC weights to PPO policy network.

    Transfers weights from BC model to PPO's policy network only.
    Value network (vf) is NOT transferred and trains from scratch.

    Layer mapping:
        BC policy_net[0] -> PPO mlp_extractor.policy_net[0]
        BC policy_net[2] -> PPO mlp_extractor.policy_net[2]
        BC action_net    -> PPO action_net

    Args:
        bc_model: Trained BCPolicyNetwork.
        ppo: MaskablePPO model to receive weights.
    """
    policy = ppo.policy
    with torch.no_grad():
        policy.mlp_extractor.policy_net[0].weight.copy_(bc_model.policy_net[0].weight)
        policy.mlp_extractor.policy_net[0].bias.copy_(bc_model.policy_net[0].bias)
        policy.mlp_extractor.policy_net[2].weight.copy_(bc_model.policy_net[2].weight)
        policy.mlp_extractor.policy_net[2].bias.copy_(bc_model.policy_net[2].bias)
        policy.action_net.weight.copy_(bc_model.action_net.weight)
        policy.action_net.bias.copy_(bc_model.action_net.bias)


def evaluate_bc_policy(
    bc_model: BCPolicyNetwork,
    n_games: int = 1000,
    n_players: int = 5,
    seed: int | None = None,
    verbose: bool = True,
):
    """Evaluate BC policy performance.

    Args:
        bc_model: Trained BCPolicyNetwork.
        n_games: Number of games to evaluate.
        n_players: Number of players per game.
        seed: Random seed for reproducibility.
        verbose: Whether to show progress.

    Returns:
        Dictionary with win_rate and avg_cards.
    """
    env = TheGameEnv(
        n_players=n_players,
        reward_per_card=0.02,
        win_reward=10.0,
        loss_penalty=0.0,
        trick_play_reward=1.0,
        distance_penalty_scale=0.003,
    )

    bc_model.eval()
    device = next(bc_model.parameters()).device
    wins = 0
    cards_list = []

    game_range = (
        tqdm(range(n_games), desc="Evaluating BC") if verbose else range(n_games)
    )

    for game_idx in game_range:
        game_seed = seed + game_idx if seed is not None else None
        obs, _ = env.reset(seed=game_seed)
        terminated = False

        while not terminated:
            mask = env.action_masks()
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                mask_tensor = torch.tensor(
                    mask, dtype=torch.bool, device=device
                ).unsqueeze(0)
                logits = bc_model(obs_tensor, mask_tensor)
                action = logits.argmax(dim=1).item()
            obs, _, terminated, _, info = env.step(action)

        cards_list.append(info.get("total_cards_played", 0))
        if info.get("victory", False):
            wins += 1

    results = {"win_rate": wins / n_games, "avg_cards": np.mean(cards_list)}

    if verbose:
        print(
            f"BC Policy - Win rate: {results['win_rate']:.2%}, Avg cards: {results['avg_cards']:.1f}"
        )

    return results


def train_bc_then_rl(
    n_demo_games: int = 10000,
    bc_epochs: int = 100,
    rl_timesteps: int = 1_000_000,
    n_players: int = 5,
    n_envs: int | None = None,
    bonus_play_threshold: int = 2,
    demo_path: Path | str | None = None,
    bc_model_path: Path | str | None = None,
    output_path: Path | str | None = None,
    seed: int | None = 42,
    verbose: int = 1,
):
    """Complete BC + RL training pipeline.

    Args:
        n_demo_games: Number of games to collect demonstrations from.
        bc_epochs: Maximum BC training epochs.
        rl_timesteps: Total RL training timesteps.
        n_players: Number of players per game.
        n_envs: Number of parallel environments for RL. Defaults to CPU count.
        bonus_play_threshold: Threshold for expert strategy.
        demo_path: Path to load/save demonstrations. Generates if not exists.
        bc_model_path: Path to load/save BC model. Trains if not exists.
        output_path: Path to save final PPO model.
        seed: Random seed for reproducibility.
        verbose: Verbosity level.

    Returns:
        Trained MaskablePPO model.
    """
    if n_envs is None:
        n_envs = os.cpu_count() or 1

    bld_dir = Path(__file__).parent.parent / "bld"
    bld_dir.mkdir(exist_ok=True)

    if demo_path is None:
        demo_path = bld_dir / "expert_demonstrations.npz"
    else:
        demo_path = Path(demo_path)

    if bc_model_path is None:
        bc_model_path = bld_dir / "bc_policy.pt"
    else:
        bc_model_path = Path(bc_model_path)

    if output_path is None:
        output_path = bld_dir / "bc_rl_100M_final.zip"
    else:
        output_path = Path(output_path)

    if verbose:
        print("=" * 60)
        print("Phase 1: Generate Expert Demonstrations")
        print("=" * 60)

    if demo_path.exists():
        if verbose:
            print(f"Loading existing demonstrations from {demo_path}")
    else:
        generate_expert_demonstrations(
            n_games=n_demo_games,
            n_players=n_players,
            bonus_play_threshold=bonus_play_threshold,
            output_path=demo_path,
            seed=seed,
            verbose=bool(verbose),
        )
    observations, actions, action_masks = load_expert_demonstrations(demo_path)

    if verbose:
        print(f"Loaded {len(observations)} state-action pairs")

    if verbose:
        print("\n" + "=" * 60)
        print("Phase 2: Behavioral Cloning")
        print("=" * 60)

    if bc_model_path.exists():
        if verbose:
            print(f"Loading existing BC model from {bc_model_path}")
        obs_dim = observations.shape[1]
        action_dim = action_masks.shape[1]
        bc_model = BCPolicyNetwork(obs_dim, action_dim)
        bc_model.load_state_dict(torch.load(bc_model_path, weights_only=True))
    else:
        bc_model = train_behavioral_cloning(
            observations=observations,
            actions=actions,
            action_masks=action_masks,
            epochs=bc_epochs,
            batch_size=256,
            learning_rate=3e-4,
            validation_split=0.1,
            patience=10,
            verbose=bool(verbose),
        )
        torch.save(bc_model.state_dict(), bc_model_path)
        if verbose:
            print(f"Saved BC model to {bc_model_path}")

    if verbose:
        print("\nEvaluating BC policy...")
    evaluate_bc_policy(bc_model, n_games=500, n_players=n_players, seed=seed)

    if verbose:
        print("\n" + "=" * 60)
        print("Phase 3: Weight Transfer to MaskablePPO")
        print("=" * 60)

    log_path = str(bld_dir / "bc_rl_logs")
    monitor_dir = str(bld_dir / "bc_rl_monitor")
    Path(monitor_dir).mkdir(exist_ok=True)

    env = create_env(
        n_players=n_players,
        reward_config=SPARSE_REWARDS,
        n_envs=n_envs,
        log_dir=monitor_dir,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )

    ppo = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        tensorboard_log=log_path,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=5e-5,
        n_steps=2048,
        batch_size=256,
        n_epochs=5,
        clip_range=0.1,
    )

    load_bc_weights_into_ppo(bc_model, ppo)
    if verbose:
        print("Transferred BC weights to PPO policy network")

    if verbose:
        print("\n" + "=" * 60)
        print("Phase 4: RL Fine-Tuning")
        print("=" * 60)
        print(f"Training for {rl_timesteps:,} timesteps with {n_envs} parallel envs")

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000_000 // n_envs,
        save_path=str(bld_dir / "bc_rl_checkpoints"),
        name_prefix="bc_rl",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = [GameMetricsCallback(verbose=verbose), checkpoint_callback]
    ppo.learn(total_timesteps=rl_timesteps, callback=callbacks)

    ppo.save(output_path)
    if verbose:
        print(f"\nSaved final model to {output_path}")

    return ppo


def main():
    """Entry point for BC + RL training script."""
    train_bc_then_rl(
        n_demo_games=10000,
        bc_epochs=100,
        rl_timesteps=100_000_000,
        n_players=5,
        verbose=1,
    )


if __name__ == "__main__":
    main()
