# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project Overview

Monte Carlo simulation framework for analyzing strategies in "The Game" (a cooperative
card game with 98 numbered cards and four stacks). The project tests different playing
strategies, evaluates shuffle quality effects on win rates, and experiments with Gemini
AI as a player.

## Commands

```bash
# Install dependencies
pixi install

# Run all simulations via pytask
pixi run pytask

# Run individual simulations
pixi run python src/simulate_strategies.py
pixi run python src/simulate_shuffle_quality.py
pixi run python src/simulate_gemini_thinking.py

# Generate plots from existing results
pixi run python src/generate_plots.py

# Run tests
pixi run pytest tests/

# Run a single test file
pixi run pytest tests/test_game_setup.py
```

## Architecture

### Core Components

- **`src/utils.py`**: Foundation layer with `Stack` class (pre-allocated O(1)
  operations), `GameOverError`, Gemini API integration, and `_play_to_stack()` helper
- **`src/game_setup.py`**: Game engine with `run_game()` and `run_simulation()` - takes
  a strategy callable with signature
  `(player, stacks, remaining_deck) -> (player, stacks)`
- **`src/strategies.py`**: Strategy implementations - `bonus_play_strategy`
  (algorithmic) and `gemini_strategy` (AI-powered)

### Stack Indices Convention

Throughout the codebase, stacks are indexed as:

- 0, 1: Decreasing stacks (start at 99, play lower cards)
- 2, 3: Increasing stacks (start at 1, play higher cards)

### Strategy Pattern

Strategies are passed to `run_game()` as callables. Use `functools.partial` to
pre-configure parameters:

```python
from functools import partial

strategy = partial(bonus_play_strategy, bonus_play_threshold=2)
```

### Test Configuration

Tests use `conftest.py` to add `src/` to the Python path. Run tests from the project
root.

## Gemini Configuration

Set `GEMINI_API_KEY` in `.env` file. The model is configured via `GEMINI_MODEL`
environment variable (defaults to `gemini-3-flash-preview`).

## Output

All generated outputs (plots, parquet files) go to `bld/` directory.

## Reinforcement Learning Training Plan

### Overview

Train a neural network via RL to play the cooperative multi-player game. Training
happens from the perspective of a single player, but all players share the same policy
(centralized training with parameter sharing).

### Recommended Method: PPO (Proximal Policy Optimization)

- Stable training with clipped objective
- Works well for discrete action spaces
- Widely used, well-documented

### Packages

```bash
pixi add gymnasium stable-baselines3 sb3-contrib
```

| Package           | Purpose                                     |
| ----------------- | ------------------------------------------- |
| gymnasium         | Standard RL environment interface           |
| stable-baselines3 | PPO implementation (PyTorch-based)          |
| sb3-contrib       | MaskablePPO for handling invalid card plays |

### Key RL Concepts to Study

**Foundations (read first):**

1. Markov Decision Process (MDP) - state, action, reward, transition
1. Value functions - V(s) and Q(s,a), Bellman equations
1. Policy gradient theorem
1. Actor-Critic architecture
1. Advantage estimation (GAE)

**Practical for this game:** 6. Action masking - block illegal card plays 7. Reward
shaping - intermediate rewards beyond win/loss 8. Centralized training with
decentralized execution (CTDE) - train one policy, all players use it

**Resource:** Spinning Up in Deep RL (OpenAI) - https://spinningup.openai.com/

### Multi-Player Environment Design

```python
class TheGameEnv(gym.Env):
    """Multi-player cooperative environment for The Game.

    Training perspective: single player's turn.
    All players share the same policy (parameter sharing).
    Other players' actions simulated using the same policy.
    """

    def __init__(self, n_players=3, hand_size=6):
        self.n_players = n_players
        self.hand_size = hand_size

        # Action: card index × stack index
        self.action_space = spaces.Discrete(hand_size * 4)

        # State: current player's hand + stack tops + deck size
        # Note: other players' hands are hidden (partial observability)
        self.observation_space = spaces.Box(
            low=0, high=99, shape=(hand_size + 4 + 1,), dtype=np.float32
        )

    def step(self, action):
        # 1. Execute current player's action
        # 2. Simulate other players using same policy
        # 3. Return to current player's next turn
        ...

    def action_masks(self):
        # Boolean array: which (card, stack) combinations are legal
        ...
```

### State Representation

| Component      | Shape | Description                  |
| -------------- | ----- | ---------------------------- |
| Player hand    | (6,)  | Card values (0 = empty slot) |
| Stack tops     | (4,)  | Current top of each pile     |
| Deck remaining | (1,)  | Cards left to draw           |

**Note:** Other players' hands are hidden (partial observability), reflecting real
gameplay.

### Reward Design

| Strategy        | Description                    |
| --------------- | ------------------------------ |
| Per-card reward | +0.01 per successful card play |
| Win bonus       | +1.0 on victory                |
| Loss penalty    | -0.5 on game over              |

```python
reward = 0.01  # Each card played
if victory:
    reward += 1.0
elif game_over:
    reward -= 0.5
```

### Training Script

```python
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from game_env import TheGameEnv

env = TheGameEnv(n_players=3)

model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log="./bld/rl_logs/")

model.learn(total_timesteps=500_000)
model.save("bld/the_game_ppo")
```

### Files to Create

| File                     | Purpose                       |
| ------------------------ | ----------------------------- |
| `src/game_env.py`        | Gymnasium environment wrapper |
| `src/train_rl.py`        | Training script               |
| `tests/test_game_env.py` | Environment validation tests  |

### Verification

1. `check_env(TheGameEnv())` - validate Gymnasium compatibility
1. Manual test with random actions
1. Short training run (1000 steps) - verify no crashes
1. Compare trained agent vs `bonus_play_strategy` win rate
1. Monitor training via TensorBoard: `pixi run tensorboard --logdir bld/rl_logs/`

## RL Reward Tuning Experiments

### Baseline Performance

| Strategy            | Players | Win Rate |
| ------------------- | ------- | -------- |
| bonus_play_strategy | 3       | 1.4%     |
| bonus_play_strategy | 5       | 4.4%     |

### Grid Search Results (300k steps, 5 players, 500 eval games)

Tested 9 reward configurations with simplified reward shaping (disabled progress_reward,
stack_health, phase_multiplier).

| Config                | Win% | Avg Cards | reward_per_card | win_reward | loss_penalty | trick_play | dist_penalty |
| --------------------- | ---- | --------- | --------------- | ---------- | ------------ | ---------- | ------------ |
| trick_and_distance    | 0%   | 63.2      | 0.05            | 10.0       | 0.0          | 1.0        | 0.005        |
| with_loss_penalty     | 0%   | 62.8      | 0.05            | 10.0       | 1.0          | 1.0        | 0.005        |
| high_card_reward      | 0%   | 62.2      | 0.10            | 10.0       | 0.0          | 1.0        | 0.005        |
| low_card_reward       | 0%   | 59.5      | 0.01            | 10.0       | 0.0          | 1.0        | 0.005        |
| very_high_card_reward | 0%   | 59.1      | 0.20            | 10.0       | 0.0          | 1.0        | 0.005        |
| high_both_terminal    | 0%   | 58.3      | 0.05            | 50.0       | 2.0          | 1.0        | 0.005        |
| high_win_bonus        | 0%   | 58.2      | 0.05            | 50.0       | 0.0          | 1.0        | 0.005        |
| simple_baseline       | 0%   | 56.2      | 0.05            | 10.0       | 0.0          | 0.0        | 0.000        |
| with_trick_bonus      | 0%   | 33.7      | 0.05            | 10.0       | 0.0          | 1.0        | 0.000        |

**Key findings:**

- All configs achieved 0% win rate (vs 4.4% baseline) - 300k steps insufficient
- `trick_and_distance` played most cards (63.2/98)
- `with_trick_bonus` alone hurt performance badly (33.7 cards) - trick bonus without
  distance penalty causes poor play
- Distance penalty improves card count significantly

### Best Known Configuration (84 cards avg, 1% win rate)

Achieved in commit `fd61005` with 2M timesteps training.

**Environment Configuration:**

```python
TheGameEnv(
    n_players=5,  # 5 players (easier than 3)
    reward_per_card=0.02,
    win_reward=10.0,
    loss_penalty=0.0,  # No penalty for losing
    trick_play_reward=1.0,
    distance_penalty_scale=0.003,
    progress_reward_scale=3.0,  # Reward partial game completion
    stack_health_scale=0.0,  # Not used
    phase_multiplier_scale=0.0,  # Not used
)
```

**Observation Space (17 features):**

- Hand cards (6)
- Stack tops (4)
- Stack gaps (4)
- Deck remaining, cards played this turn, min cards required (3)
- Note: Simpler than current (no other_hand_sizes, total_progress, hand_stats)

**PPO Hyperparameters:**

```python
MaskablePPO(
    gamma=0.99,  # Discount factor
    gae_lambda=0.95,
    ent_coef=0.02,  # Fixed (no decay)
    learning_rate=3e-4,  # Fixed (no schedule)
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    clip_range=0.2,
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
)
```

**Training Setup:**

- Total timesteps: 2,000,000 (2M)
- Parallel environments: CPU count (typically 8-12)
- Training time: ~30-60 minutes on modern CPU

**Results:**

- Average cards played: ~84 / 98 (86%)
- Win rate: 1% (vs 4.4% baseline strategy)
- Random baseline: ~14 cards

**Key insights:**

- No loss penalty allows agent to explore risky plays
- Progress reward (3.0 scale) crucial for learning partial success
- Simpler observation space may improve learning efficiency
- Fixed hyperparameters (vs schedules) worked well
- 5 players makes game easier due to more cards per hand
