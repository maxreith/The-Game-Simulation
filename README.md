# The Game Simulation

Training RL agents on [The Game](https://boardgamegeek.com/boardgame/173090/the-game), a
cooperative card game where players work together to play cards onto four shared stacks.

## The Game Rules

- 98 cards numbered 2-99
- Four stacks: two ascending (start at 1), two descending (start at 99)
- Cards must be played in the correct direction, except for the "backwards trick":
  playing exactly 10 higher/lower resets the stack
- Each turn: play at least 2 cards (or 1 if deck is empty), then draw back to hand size
- Win condition: all cards played; lose condition: cannot make a legal play

## Training Agents

I use gymnasium and sblines3 for training. Masked PPO.

`Game_env.py` defines the environment. Agents' observation space includes own hand,
stack tops, the number of cards remaining in the deck, how many cards the agents has
left to play this turn and how many he has played already, total progress. Importantly,
an agent does not get to see the other agents' hands, as that would be against the
rules!

Agents are trained from an individual perspective, but they are rewarded for collective
outcomes.

I employ two ways to train agents. I train agents on games with n players.

1. Simply training an agent

Explain reward structure.

Then show training results. My hypothesis is that training will not go well.

2. BC + RL

For this, I first coded up an expert system called bonus_play_strategy, saved in
strategies.py. It's a pretty simple system: it always plays the card with the minimum
distance to the current stack tops until it reached the minimum number of cards to play,
and only plays additional cards if the distance is below a bonus_play threshold.

The expert is able to get decent winrates of up to 5% on the game. I ran 10,000
simulations for each combination of player count and bonus play threshold.

![Strategy Evaluation](bld/strategy_evaluation.png)

The highest winrate across all combinations is achieved in a 5 player game, with a
threshold of 2.

The way the reinforcement learning agent is trained here is to first 'clone' the expert
system into a neural network, and then finetune the neural network using RL.

Show training results of BC + RL agent.

## Further Analysis

### Does it matter how well the deck is shuffled?

I used to disagree with my office mates over how well we'd have to shuffle the deck
before setting up a new game. Using a custom cut-based shuffle algorithm, I simulate how
shuffle quality affects win rates. I ran 1,000 games per shuffle quality, using the
optimal settings (5 players, bonus play threshold of 2).

![Shuffle Quality Evaluation](bld/shuffle_evaluation.png)

**Findings:**

- Poorly shuffled decks have dramatically higher win rates
- With proper shuffling (50+ iterations), win rates stabilize around 5%

### Can AI play this game?

The answer is no. I wanted to test Gemini 3 on this, but couldn't get it to win a single
game. It keeps making invalid moves. I guess there is no training data for this game.
But does Gemini at least perform better when you increase the thinking level? The plot
below shows the average turn at which Gemini lost, for various thinking levels. I ran 3
games per thinking level (because compute is not free, and high thinking is slow).

![Gemini Thinking Levels](bld/gemini_thinking.png)

**Findings:**

- Higher thinking levels survive more turns
- Even with extended thinking, the AI struggles with the game

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
