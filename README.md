# The Game Simulation

## Results

![Bonus Play Strategy Results](plots/bonus_play_results.png)

## Gemini Thinking Level Simulation

To run the Gemini thinking level simulation:

```bash
pixi run python src/simulate_gemini_thinking.py
```

Configure simulation parameters at the top of `src/simulate_gemini_thinking.py`:

- `THINKING_LEVELS`: List of thinking levels to test (e.g.,
  `["minimal", "low", "medium", "high"]`)
- `N_GAMES_PER_LEVEL`: Number of games per level (e.g., `3`)

The simulation runs games in parallel across thinking levels to reduce total runtime.
Results are saved to `bld/gemini_thinking_results.parquet` and total simulation time is
printed upon completion.
