"""
Quick test to verify the simulation works before running full simulations.
"""

from simulate_shuffle_quality import find_optimal_parameters, evaluate_shuffle_qualities

# Test reading optimal parameters from parquet
optimal = find_optimal_parameters("simulation_results.parquet")
print(
    f"Optimal parameters: n_players={optimal['n_players']}, threshold={optimal['bonus_play_threshold']}, win_rate={optimal['win_rate']:.2%}"
)

# Quick test of shuffle qualities with small numbers
print("\nTesting shuffle qualities with n_games=5...")
shuffle_df = evaluate_shuffle_qualities(optimal, n_games=5)
print("Shuffle test completed!")
print(f"\nResults shape: {shuffle_df.shape}")
print(f"Win rates: {shuffle_df['win_rate'].tolist()}")
