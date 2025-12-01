from game_strategies import simple_game_strategy
from simulation import run_simulation


results = run_simulation(simple_game_strategy, n_games=10, n_players=3)

print(f"Win rate: {results['win_rate']*100:.1f}%")
print(f"Victories: {len(results['victories'])}")
print(f"Losses: {len(results['losses'])}")