# The Game RL

Training RL agents on [The Game](https://boardgamegeek.com/boardgame/173090/the-game), a
cooperative card game where players work together to play cards onto four shared stacks.
I play this game with my mates a lot, so I had to do this.

## The Game Rules

- 98 cards numbered 2-99
- Four stacks: two ascending (start at 1), two descending (start at 99)
- Cards must be played in the correct direction, except for the "backwards trick":
  playing exactly 10 higher/lower resets the stack
- Each turn: play at least 2 cards (or 1 if deck is empty), then draw back to hand size
- Win condition: all cards played; lose condition: cannot make a legal play

## Training Agents

I train RL agents using masked proximal policy optimization (PPO).

PPO methods use actor-critic architecture, where the actor refers to the policy
$\\pi(a|s) \\rightarrow [0,1]$, the probability of taking action $a$ in state $s$, and
the critic parametrized by $\\theta$ estimates the approximate value function for the
current policy: $Q(s, a; \\theta) \\approx Q^{\\pi}(s, a)$.

As is natural in this game, agents are rewarded for collective outcomes, but only get to
observe their own private information. I use `gymnasium` to define the training
environment. Agents' observation space includes the cards in their own hand, the cards
on top of the four stacks, the number of cards remaining in the deck, the number of
cards the agent has left to play in the current turn and the number of cards the agent
has played already in the current turn and finally the number of cards each other player
has left on their hand.

I use `stable_baselines3` for training. All agents are trained on games with 5 players.
In the following, I will outline two training regimes.

### Pure Reinforcement Learning

First, I train an agent with a **sparse reward structure** that only provides terminal
feedback:

| Parameter      | Value |
| -------------- | ----- |
| `win_reward`   | 100.0 |
| `loss_penalty` | 0.5   |

These rewards seem natural: just tell the agent what the goal is and let it figure out
how to get there. But after 100 million training steps with MaskablePPO, the agent
performs rather poorely.

| Metric           | Value |
| ---------------- | ----- |
| Win rate         | 3.5%  |
| Avg cards played | 86.5  |

From experience, I can say that a 3.5% win rate is very low. The poor performance is
probably because the reward signal is sparse. A game with 86 cards played, contains of
~120 steps. Yet, the agent receives no signal during most of the gameplay, but is only
rewarded or punished on the final step of the game.

To improve the win rate, I want to guide learning more. Thus, I add shaped rewards to
guide learning:

| Parameter                | Value | Purpose                               |
| ------------------------ | ----- | ------------------------------------- |
| `reward_per_card`        | 0.02  | Encourage playing cards               |
| `win_reward`             | 100.0 | Terminal win bonus                    |
| `loss_penalty`           | 0.5   | Terminal loss penalty                 |
| `trick_play_reward`      | 1.0   | Bonus for backwards trick plays (±10) |
| `distance_penalty_scale` | 0.003 | Penalty for large gaps (distance > 5) |

The distance penalty is quadratic: `penalty = 0.003 * (distance - 5)²` for plays where
the card is more than 5 away from the stack top. This discourages wasteful plays.

After 100 million training steps, the win rate has nearly tripled to 8.9%!

| Metric           | Value |
| ---------------- | ----- |
| Win rate         | 8.9%  |
| Avg cards played | 88.2  |

Can I improve the win rate even further?

### 2. Behaviorial Cloning + RL Finetuning

With a single game spanning more than 100 training steps, I was concerned that most of
the 100M training budget would be spent on the agent learning basic game intuitions —
avoid large gaps, don't waste cards — leaving little capacity to learn more nuanced
strategy. To bootstrap the agent with basic game knowledge, I built a simple expert for
the agent to imitate before fine-tuning with RL.

#### Behavioral Cloning

I first built a simple expert called bonus_play_strategy. The strategy is
straightforward: always play the card closest to any stack top until the minimum number
of cards has been played, then continue playing additional cards only if they are within
a set distance threshold of a stack top.

What win rates does the expert achieve? The graph below shows results from 10,000
simulations for each combination of player count/bonus play threshold combination. The
expert is able to achieve winrates of 5%. The highest win rate is in a game with 5
players and a bonus play threshold of 2.

![Strategy Evaluation](bld/strategy_evaluation.png)

I "cloned" the expert into a neural network via supervised learning. I collect 10,000
games of expert play (~1.27M state-action pairs), then train a policy network to predict
the expert's action given each game state.

The BC policy achieves **98% validation accuracy** on held-out data. When evaluated in
actual gameplay, it slightly outperforms the expert it was trained on:

| Agent  | Win Rate |
| ------ | -------- |
| Expert | 4.5%     |
| BC     | 4.7%     |

#### RL Fine-Tuning

To fine tune the agent, I initialize MaskablePPO with the BC weights and fine-tune the
agent with RL. I use more conservative hyperparameters for training:

| Parameter       | Value | Rationale                       |
| --------------- | ----- | ------------------------------- |
| `learning_rate` | 5e-5  | 6x lower than pure RL           |
| `clip_range`    | 0.1   | Tighter clipping for stability  |
| `ent_coef`      | 0.01  | Less exploration (already good) |
| `n_epochs`      | 5     | Fewer updates per batch         |

RL fine-tuning uses sparse rewards (win/loss only) to let the agent discover
improvements beyond the expert's strategy.

| Training Steps | Win Rate | Avg Cards |
| -------------- | -------- | --------- |
| 0 (BC only)    | 5.5%     | 87.0      |
| 100M           | 14.4%    | 91.4      |

BC+RL reaches **14.4% win rate** after 100M steps, 3x the expert baseline and ~1.6x pure
RL with shaped rewards!

## Further Analysis

### Does it matter how well the deck is shuffled?

My mates and I keep disagreeing over how much to shuffle the deck before setting up a
new game. Well, how does shuffling affect win rates? Using a custom cut-based shuffle
algorithm, different shuffle qualities at 1000 games each, using the simple expert
system (`bonus_play_strategy`) with optimal settings (5 players, bonus play threshold of
2).

![Shuffle Quality Evaluation](bld/shuffle_evaluation.png)

You have to properly shuffle your deck!

### Can AI play this game?

I wanted to test Gemini 3 on this, but couldn't get it to win a single game. It keeps
making invalid moves, probably because there is little training data on how to play this
card game. But does Gemini at least perform better when you increase the thinking level?
The plot below shows the average turn at which Gemini lost, for various thinking levels.
I ran 3 games per thinking level, which already cost roughly $5 in API calls.

![Gemini Thinking Levels](bld/gemini_thinking.png)

Higher thinking leads to surviving more turns. My mates should give it a try ;)

## Installation

```bash
pixi install
```

## Usage

Run simulations:

```bash
pixi run python src/simulate_strategies.py
pixi run python src/simulate_shuffle_quality.py
pixi run python src/simulate_gemini_thinking.py
```

Generate plots from existing results:

```bash
pixi run python src/generate_plots.py
```

Train RL agents:

```bash
pixi run python src/train_rl.py
pixi run python src/train_bc_rl.py
```

Evaluate trained models:

```bash
pixi run python src/evaluate_rl.py
```

## Project Structure

```
src/
├── game_setup.py              # Core game mechanics
├── game_env.py                # Gymnasium environment for RL training
├── strategies.py              # Playing strategies (bonus play, Gemini)
├── utils.py                   # Stack implementation, Gemini API integration
├── train_rl.py                # Pure RL training script
├── train_bc_rl.py             # Behavioral cloning + RL training
├── evaluate_rl.py             # Model evaluation and comparison
├── generate_expert_data.py    # Generate expert demonstrations for BC
├── generate_example_games.py  # Generate example game logs
├── generate_plots.py          # Visualization generation
├── plot_training_curves.py    # Plot training metrics from logs
├── simulate_strategies.py     # Strategy comparison simulation
├── simulate_shuffle_quality.py    # Shuffle quality analysis
└── simulate_gemini_thinking.py    # Gemini thinking level tests

tests/                         # Unit tests
bld/                           # Generated outputs (plots, results, models)
```

## Configuration

For Gemini simulations, set your API key in `.env`:

```
GEMINI_API_KEY=your_key_here
```
