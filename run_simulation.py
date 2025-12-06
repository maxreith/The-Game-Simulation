import itertools
from pathlib import Path
import pandas as pd
from game_strategies import bonus_play_strategy, simple_game_strategy
from game_setup import run_simulation

# Define your parameter grid
param_grid = {
    "strategy": [bonus_play_strategy],  # Add more strategies as you create them
    "n_players": [2, 3, 4, 5, 6],
    "bonus_play_threshold": [0, 1, 2, 3, 4, 5, 6, 8, 10],
    "n_games": [10000],
}

# Strategy name lookup (for readable output)
strategy_names = {
    simple_game_strategy: "simple",
    bonus_play_strategy: "bonus_play",
}

# Generate all combinations
keys = param_grid.keys()
combinations = list(itertools.product(*param_grid.values()))

# Run simulations for each combination
all_results = []
for combo in combinations:
    params = dict(zip(keys, combo))
    
    results = run_simulation(
        params["strategy"], 
        n_games=params["n_games"], 
        n_players=params["n_players"],
        bonus_play_threshold=params["bonus_play_threshold"]
    )
    
    all_results.append({
        "strategy": strategy_names[params["strategy"]],
        "n_players": params["n_players"],
        "bonus_play_threshold": params["bonus_play_threshold"],
        "n_games": params["n_games"],
        "win_rate": results["win_rate"],
        "victories": len(results["victories"]),
        "losses": len(results["losses"]),
    })
    
    print(f"✓ {strategy_names[params['strategy']]}, {params['n_players']} players, bonus_threshold={params['bonus_play_threshold']}: {results['win_rate']*100:.1f}%")

# Save results
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

df = pd.DataFrame(all_results)
df.to_parquet(output_dir / "simulation_results.parquet", index=False)
print(f"\nResults saved to {output_dir / 'simulation_results.parquet'}")
print(df)