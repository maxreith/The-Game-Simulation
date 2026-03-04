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

## Reinforcement Learning

Uses MaskablePPO from `sb3-contrib` with gymnasium environment. Training from single
player perspective with parameter sharing across all players.

```bash
pixi run python src/train_rl.py
pixi run tensorboard --logdir bld/rl_logs/
```

### Post-Training Outputs

After training a model, produce:

1. **Graphs**: Run `pixi run python src/generate_plots.py` (if applicable data exists)

1. **Example game log** at `bld/{model_name}_example_game.md` with 3 example games
   showing:

   - Step-by-step gameplay from start to finish
   - At each step: player number, hand, stack tops, and the action taken
   - For losses: explain why the agent failed (which cards couldn't be played and why)

Example format for the game log:

```
## Example Game 1 (seed=42)

### Step 1: Player 1
- Hand: [12, 25, 34, 67, 78, 91]
- Stacks: Dec1=99, Dec2=99, Inc1=1, Inc2=1
- Action: Play 12 on Inc1 (top=1)

### Step 2: Player 1
- Hand: [25, 34, 67, 78, 91]
- Stacks: Dec1=99, Dec2=99, Inc1=12, Inc2=1
- Action: Play 25 on Inc1 (top=12)
...

### Final Step: Player 3 - LOSS
- Hand: [50]
- Stacks: Dec1=10, Dec2=10, Inc1=95, Inc2=95
- No valid moves: 50 cannot be played (need <10 or >95)
```

## RL Experiment Results

### Baseline

| Strategy            | Players | Win Rate      |
| ------------------- | ------- | ------------- |
| bonus_play_strategy | 3       | 1.4%          |
| bonus_play_strategy | 5       | 4.4%          |
| random              | 5       | ~14 cards avg |

### Grid Search (300k steps, 5 players, 500 eval games)

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

### Extended Training: trick_and_distance at 2M steps

| Training Steps | Avg Cards | Win Rate |
| -------------- | --------- | -------- |
| 300k           | 63.2      | 0%       |
| 2M             | 84.7      | 0%       |

Configuration:
`n_players=5, reward_per_card=0.05, win_reward=10.0, loss_penalty=0.0, trick_play_reward=1.0, distance_penalty_scale=0.005, progress_reward_scale=0.0`

### Best Configuration (2M steps, commit fd61005)

| Metric        | Value   |
| ------------- | ------- |
| Win rate      | 1%      |
| Avg cards     | 84      |
| Training time | ~35 min |

**Environment:**
`n_players=5, reward_per_card=0.02, win_reward=10.0, loss_penalty=0.0, trick_play_reward=1.0, distance_penalty_scale=0.003, progress_reward_scale=3.0`

**Observation space (17 features):** hand(6), stack_tops(4), stack_gaps(4),
deck_remaining, cards_played_this_turn, min_cards_required

**PPO:**
`gamma=0.99, gae_lambda=0.95, ent_coef=0.02, learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10, clip_range=0.2, net_arch=[256,256]`

### Simplified Rewards (20M steps)

| Metric        | Value   |
| ------------- | ------- |
| Win rate      | 4.8%    |
| Avg cards     | 87.9    |
| Training time | ~58 min |

**Reward structure:**
`reward_per_card=0.02, win_reward=100.0, loss_penalty=0.5, trick_play_reward=1.0, distance_penalty_scale=0.003, progress_reward_scale=0.0, stack_health_scale=0.0, phase_multiplier_scale=0.0`

**Key changes:** Removed progress_reward_scale, stack_health_scale, and
phase_multiplier_scale. Increased win_reward from 10 to 100. Reactivated loss_penalty at
0.5. This simplified structure with a large terminal win bonus outperforms the previous
best pure RL (1%) and matches the expert baseline (4.4%).

### Hierarchical RL (sequential card-then-stack decisions)

Breaks turn into phases: CHOOSE_CARD(0-5) → CHOOSE_STACK(0-3) → CONTINUE(0-1). Reduces
action space from 25 to 6.

Files: `src/hierarchical_game_env.py`, `src/train_hierarchical_rl.py`,
`src/evaluate_hierarchical.py`

```bash
pixi run python src/train_hierarchical_rl.py
pixi run python src/evaluate_hierarchical.py
```

Same reward config as best known. Results not yet documented.

### BC+RL (Behavioral Cloning + RL Fine-Tuning)

Two-phase approach: pre-train policy via supervised learning on expert demonstrations,
then fine-tune with RL.

Files: `src/generate_expert_data.py`, `src/train_bc_rl.py`

```bash
pixi run python src/train_bc_rl.py
```

**Results (5 players, 1000 eval games):**

| Agent                | Win Rate | Avg Cards | Training Steps |
| -------------------- | -------- | --------- | -------------- |
| Expert (bonus_play)  | 4.2%     | 87.3      | -              |
| BC-only              | 5.5%     | 87.0      | -              |
| BC+RL (full rewards) | 3.0%     | 87.5      | 1M             |
| BC+RL (simplified)   | 5.3%     | 88.0      | 2M             |
| Pure RL (simplified) | 4.8%     | 87.9      | 20M            |
| Pure RL (old best)   | 1.0%     | 84.0      | 2M             |

**Key finding:** Simplified reward shaping during RL fine-tuning preserves BC policy
quality. Setting `reward_per_card=0, progress_reward_scale=0, phase_multiplier_scale=0`
achieves 5.3% win rate, nearly matching BC-only (5.5%).

**BC training:** 98% validation accuracy, ~20 min training on 1.27M expert samples.

**RL fine-tuning hyperparams (conservative to preserve BC knowledge):**
`learning_rate=5e-5, ent_coef=0.01, clip_range=0.1, n_epochs=5`

**Simplified rewards for RL phase:**
`reward_per_card=0, win_reward=10.0, trick_play_reward=1.0, distance_penalty_scale=0.003, progress_reward_scale=0, phase_multiplier_scale=0`

### Final Comparison (5 players, 10000 eval games, seed=42)

| Model                   | Win Rate | Avg Cards |
| ----------------------- | -------- | --------- |
| Pure RL (sparse) @ 100M | 4.0%     | 86.9      |
| Pure RL (shaped) @ 100M | 8.1%     | 88.2      |
| BC+RL @ 100M            | 14.4%    | 91.4      |
| BC-only                 | 4.7%     | 87.2      |
| Baseline (bonus_play)   | 4.5%     | -         |

**Key findings:**

- BC+RL significantly outperforms pure RL at the same step count (14.4% vs 8.1% at 100M)
- BC+RL exceeds the expert baseline it was trained on (14.4% vs 4.5%)
- Shaped rewards outperform sparse rewards for pure RL (8.1% vs 4.0%)
