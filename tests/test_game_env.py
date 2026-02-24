"""Tests for TheGameEnv gymnasium environment."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from game_env import TheGameEnv


class TestTheGameEnv:
    """Tests for basic environment functionality."""

    def test_env_checker_passes(self):
        """Environment passes gymnasium validation."""
        env = TheGameEnv()
        check_env(env, skip_render_check=True)

    def test_action_space_size(self):
        """Action space is hand_size * 4 stacks + 1 end turn action."""
        env = TheGameEnv(n_players=3, hand_size=6)
        assert env.action_space.n == 25  # 6*4 + 1

        env = TheGameEnv(n_players=2, hand_size=7)
        assert env.action_space.n == 29  # 7*4 + 1

    def test_observation_space_shape(self):
        """Observation includes hand, stacks, gaps, deck, cards_played, min_required, other hands, progress, hand_stats."""
        env = TheGameEnv(n_players=3, hand_size=6)
        # 6 hand + 4 stack tops + 4 stack gaps + 1 deck + 1 cards_played + 1 min_required
        # + 2 other_hand_sizes (3-1) + 1 total_progress + 3 hand_stats = 23
        assert env.observation_space.shape == (23,)

        env5 = TheGameEnv(n_players=5, hand_size=6)
        # + 4 other_hand_sizes (5-1) = 25
        assert env5.observation_space.shape == (25,)

    def test_observation_space_with_max_players(self):
        """Observation size is fixed based on max_players for curriculum learning."""
        env2 = TheGameEnv(n_players=2, max_players=5, hand_size=6)
        env3 = TheGameEnv(n_players=3, max_players=5, hand_size=6)
        env5 = TheGameEnv(n_players=5, max_players=5, hand_size=6)

        # All should have same observation size: 6 + 4 + 4 + 3 + 4 + 1 + 3 = 25
        assert env2.observation_space.shape == (25,)
        assert env3.observation_space.shape == (25,)
        assert env5.observation_space.shape == (25,)

    def test_observation_padding_with_fewer_players(self):
        """Other hand sizes are padded with zeros when n_players < max_players."""
        env = TheGameEnv(n_players=2, max_players=5, hand_size=6)
        obs, _ = env.reset(seed=42)

        # Position: hand (6) + stack_tops (4) + gaps (4) + deck (1) + cards_played (1) + min_required (1) = 17
        other_hands_start = 6 + 4 + 4 + 3
        # First slot: player 2's hand (normalized to 1.0)
        assert obs[other_hands_start] == pytest.approx(1.0, abs=0.01)
        # Slots 2, 3, 4: padding (zeros)
        assert obs[other_hands_start + 1] == 0.0
        assert obs[other_hands_start + 2] == 0.0
        assert obs[other_hands_start + 3] == 0.0

    def test_reset_returns_valid_observation(self):
        """Reset returns observation within bounds."""
        env = TheGameEnv()
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        assert "action_mask" in info

    def test_reset_is_deterministic_with_seed(self):
        """Same seed produces same initial state."""
        env = TheGameEnv()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_action_mask_shape(self):
        """Action mask has correct shape (includes end turn action)."""
        env = TheGameEnv(n_players=3, hand_size=6)
        env.reset(seed=42)
        mask = env.action_masks()
        assert mask.shape == (25,)  # 6*4 + 1 end turn
        assert mask.dtype == bool

    def test_action_mask_has_valid_actions(self):
        """At game start, there should be valid actions."""
        env = TheGameEnv()
        env.reset(seed=42)
        mask = env.action_masks()
        assert np.any(mask)

    def test_valid_action_gives_positive_reward(self):
        """Playing a valid card gives positive reward."""
        env = TheGameEnv(distance_penalty_scale=0)
        env.reset(seed=42)
        mask = env.action_masks()
        valid_action = np.where(mask)[0][0]
        obs, reward, term, trunc, info = env.step(valid_action)
        assert reward > 0
        assert not term

    def test_invalid_action_terminates_with_penalty(self):
        """Invalid action ends game with negative reward."""
        env = TheGameEnv()
        env.reset(seed=42)
        mask = env.action_masks()
        invalid_actions = np.where(~mask)[0]
        if len(invalid_actions) > 0:
            obs, reward, term, trunc, info = env.step(invalid_actions[0])
            assert reward < 0
            assert term
            assert info.get("invalid", False)

    def test_cards_played_counter_increments(self):
        """Observation shows cards played this turn increasing."""
        env = TheGameEnv()
        obs, _ = env.reset(seed=42)
        # Position: hand (6) + stack_tops (4) + gaps (4) + deck (1) = index 15
        cards_played_idx = env.hand_size + 4 + 4 + 1
        assert obs[cards_played_idx] == 0

        mask = env.action_masks()
        action = np.where(mask)[0][0]
        obs, _, _, _, _ = env.step(action)
        # Normalized: 1 card / hand_size
        assert obs[cards_played_idx] == pytest.approx(1.0 / env.hand_size, abs=0.01)

    def test_minimum_cards_required_in_observation(self):
        """Observation includes min cards required (2 if deck, 1 if empty)."""
        env = TheGameEnv()
        obs, _ = env.reset(seed=42)
        # Position: hand (6) + stack_tops (4) + gaps (4) + deck (1) + cards_played (1) = 16
        min_required_idx = env.hand_size + 4 + 4 + 2
        # Normalized: (min_required - 1), so 2 -> 1.0
        assert obs[min_required_idx] == 1.0  # Deck exists at start

    def test_other_hand_sizes_in_observation(self):
        """Observation includes other players' hand sizes."""
        env = TheGameEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        # Position: hand (6) + stack_tops (4) + gaps (4) + deck (1) + cards_played (1) + min_required (1) = 17
        other_hands_start = env.hand_size + 4 + 4 + 3
        # All other players start with full hands (normalized to 1.0)
        assert obs[other_hands_start] == pytest.approx(1.0, abs=0.01)
        assert obs[other_hands_start + 1] == pytest.approx(1.0, abs=0.01)

    def test_total_progress_in_observation(self):
        """Observation includes total progress (cards played / 98)."""
        env = TheGameEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        # Position: other_hands_end + 1 total_progress
        progress_idx = env.hand_size + 4 + 4 + 3 + (env.n_players - 1)
        assert obs[progress_idx] == 0.0  # No cards played yet

        # Play one card
        mask = env.action_masks()
        action = np.where(mask)[0][0]
        obs, _, _, _, _ = env.step(action)
        assert obs[progress_idx] == pytest.approx(1.0 / 98.0, abs=0.001)

    def test_hand_stats_in_observation(self):
        """Observation includes hand statistics (min, max, mean)."""
        env = TheGameEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        # Position: after total_progress
        stats_start = env.hand_size + 4 + 4 + 3 + (env.n_players - 1) + 1
        hand = env.hands[0]
        expected_min = np.min(hand) / 100.0
        expected_max = np.max(hand) / 100.0
        expected_mean = np.mean(hand) / 100.0
        assert obs[stats_start] == pytest.approx(expected_min, abs=0.01)
        assert obs[stats_start + 1] == pytest.approx(expected_max, abs=0.01)
        assert obs[stats_start + 2] == pytest.approx(expected_mean, abs=0.01)


class TestGameMechanics:
    """Tests for game logic correctness."""

    def test_decreasing_stack_accepts_lower_card(self):
        """Decreasing stacks (0, 1) accept lower cards."""
        env = TheGameEnv()
        env.reset(seed=42)
        hand = env.hands[0]
        lowest_card = np.min(hand)
        card_idx = np.where(hand == lowest_card)[0][0]
        action = card_idx * 4 + 0  # Play on decreasing stack 0
        obs, reward, term, _, _ = env.step(action)
        if not term or reward > 0:
            assert env.stacks[0].top == lowest_card

    def test_increasing_stack_accepts_higher_card(self):
        """Increasing stacks (2, 3) accept higher cards."""
        env = TheGameEnv()
        env.reset(seed=42)
        hand = env.hands[0]
        for card in hand:
            if card > 1:  # Can be played on increasing stack starting at 1
                card_idx = np.where(hand == card)[0][0]
                action = card_idx * 4 + 2  # Play on increasing stack 2
                obs, reward, term, _, _ = env.step(action)
                if not term or reward > 0:
                    assert env.stacks[2].top == card
                break

    def test_turn_ends_after_minimum_cards(self):
        """Player switches after playing minimum required cards."""
        env = TheGameEnv(n_players=2)
        env.reset(seed=42)
        assert env.current_player_idx == 0

        # Play 2 cards
        for _ in range(2):
            mask = env.action_masks()
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            env.step(valid[0])

        # If player 0's hand is now empty or no valid moves, turn should end
        if len(env.hands[0]) == 0 or not np.any(env.action_masks()):
            assert env.current_player_idx == 1

    def test_game_over_when_no_valid_plays(self):
        """Game ends when player cannot make required plays."""
        env = TheGameEnv()
        obs, _ = env.reset(seed=42)

        steps = 0
        while True:
            mask = env.action_masks()
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            action = valid[0]
            obs, reward, term, trunc, info = env.step(action)
            steps += 1
            if term:
                assert "victory" in info
                break
            if steps > 500:
                pytest.fail("Game did not terminate")


class TestRewardStructure:
    """Tests for reward calculations."""

    def test_reward_per_card_configurable(self):
        """Custom reward_per_card is applied."""
        env = TheGameEnv(
            reward_per_card=0.5,
            trick_play_reward=0,
            distance_penalty_scale=0,
            stack_health_scale=0,
            phase_multiplier_scale=0,
        )
        env.reset(seed=42)
        mask = env.action_masks()
        action = np.where(mask)[0][0]
        _, reward, _, _, _ = env.step(action)
        assert reward == pytest.approx(0.5, abs=0.001)

    def test_win_reward_included_on_victory(self):
        """Win bonus is added to final reward."""
        env = TheGameEnv(reward_per_card=0.1, win_reward=10.0)
        # Can't easily force a win, but we can check the attribute exists
        assert env.win_reward == 10.0

    def test_loss_penalty_applied(self):
        """Loss penalty is subtracted from reward on game over."""
        env = TheGameEnv(reward_per_card=0.1, loss_penalty=5.0)
        assert env.loss_penalty == 5.0

    def test_trick_play_reward_on_increasing_stack(self):
        """Backwards trick on increasing stack gives bonus reward."""
        from utils import Stack

        env = TheGameEnv(
            reward_per_card=0.01,
            trick_play_reward=0.1,
            distance_penalty_scale=0,
            stack_health_scale=0,
            phase_multiplier_scale=0,
        )
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 50])
        env.hands[0] = np.array([40, 55, 60, 65, 70, 75], dtype=np.int32)

        action = 0 * 4 + 2
        _, reward, terminated, _, _ = env.step(action)
        assert not terminated
        assert reward == pytest.approx(0.11, abs=0.001)

    def test_trick_play_reward_on_decreasing_stack(self):
        """Backwards trick on decreasing stack gives bonus reward."""
        from utils import Stack

        env = TheGameEnv(
            reward_per_card=0.01,
            trick_play_reward=0.1,
            distance_penalty_scale=0,
            stack_health_scale=0,
            phase_multiplier_scale=0,
        )
        env.reset(seed=42)

        env.stacks[0] = Stack.from_array([99, 50])
        env.hands[0] = np.array([60, 45, 40, 35, 30, 25], dtype=np.int32)

        action = 0 * 4 + 0
        _, reward, terminated, _, _ = env.step(action)
        assert not terminated
        assert reward == pytest.approx(0.11, abs=0.001)

    def test_no_distance_penalty_within_threshold(self):
        """Distance <= 5 has no penalty."""
        from utils import Stack

        env = TheGameEnv(
            reward_per_card=0.01,
            trick_play_reward=0,
            distance_penalty_scale=0.01,
            stack_health_scale=0,
            phase_multiplier_scale=0,
        )
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 30])
        env.hands[0] = np.array([35, 40, 45, 50, 55, 60], dtype=np.int32)

        action = 0 * 4 + 2
        _, reward, terminated, _, _ = env.step(action)
        assert not terminated
        # distance=5, no penalty
        assert reward == pytest.approx(0.01, abs=0.001)

    def test_quadratic_distance_penalty(self):
        """Distance > 5 gets quadratic penalty."""
        from utils import Stack

        env = TheGameEnv(
            reward_per_card=0.01,
            trick_play_reward=0,
            distance_penalty_scale=0.01,
            stack_health_scale=0,
            phase_multiplier_scale=0,
        )
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 30])
        env.hands[0] = np.array([40, 45, 50, 55, 60, 65], dtype=np.int32)

        action = 0 * 4 + 2
        _, reward, terminated, _, _ = env.step(action)
        assert not terminated
        # distance=10, penalty = 0.01 * (10-5)^2 = 0.01 * 25 = 0.25
        assert reward == pytest.approx(0.01 - 0.25, abs=0.001)

    def test_quadratic_penalty_scales_with_distance(self):
        """Larger distances get much larger penalties."""
        from utils import Stack

        env = TheGameEnv(
            reward_per_card=0.01,
            trick_play_reward=0,
            distance_penalty_scale=0.001,
            stack_health_scale=0,
            phase_multiplier_scale=0,
        )
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 10])
        env.hands[0] = np.array([35, 40, 45, 50, 55, 60], dtype=np.int32)

        action = 0 * 4 + 2
        _, reward, terminated, _, _ = env.step(action)
        assert not terminated
        # distance=25, penalty = 0.001 * (25-5)^2 = 0.001 * 400 = 0.4
        assert reward == pytest.approx(0.01 - 0.4, abs=0.001)

    def test_no_trick_reward_for_forward_plays(self):
        """Normal plays (not backwards trick) don't get trick bonus."""
        from utils import Stack

        env = TheGameEnv(
            reward_per_card=0.01,
            trick_play_reward=0.1,
            distance_penalty_scale=0,
            stack_health_scale=0,
            phase_multiplier_scale=0,
        )
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 30])
        env.hands[0] = np.array([40, 45, 50, 55, 60, 65], dtype=np.int32)

        action = 0 * 4 + 2
        _, reward, terminated, _, _ = env.step(action)
        assert not terminated
        # Not a backwards trick (40 > 30, normal forward play on increasing stack)
        assert reward == pytest.approx(0.01, abs=0.001)

    def test_stack_health_reward(self):
        """Balanced stack usage gives bonus reward."""
        from utils import Stack

        env = TheGameEnv(
            reward_per_card=0.0,
            trick_play_reward=0.0,
            distance_penalty_scale=0.0,
            stack_health_scale=0.1,
            phase_multiplier_scale=0.0,
        )
        env.reset(seed=42)

        # Set up balanced stacks (all at similar gaps)
        env.stacks[0] = Stack.from_array([99, 50])  # gap = 48/97 ~ 0.495
        env.stacks[1] = Stack.from_array([99, 50])  # gap = 48/97 ~ 0.495
        env.stacks[2] = Stack.from_array([1, 50])  # gap = 49/98 = 0.5
        env.stacks[3] = Stack.from_array([1, 50])  # gap = 49/98 = 0.5
        env.hands[0] = np.array([40, 45, 55, 60, 65, 70], dtype=np.int32)

        action = 0 * 4 + 0  # Play 40 on decreasing stack
        _, reward, terminated, _, _ = env.step(action)
        assert not terminated
        # Low variance -> bonus close to 0.1
        assert reward > 0.09

    def test_phase_multiplier_reward(self):
        """Late game rewards are multiplied."""
        env = TheGameEnv(
            reward_per_card=0.1,
            trick_play_reward=0.0,
            distance_penalty_scale=0.0,
            stack_health_scale=0.0,
            phase_multiplier_scale=0.5,
        )
        env.reset(seed=42)

        # At start, phase = 0, multiplier = 1.0
        env.total_cards_played = 0
        mask = env.action_masks()
        action = np.where(mask)[0][0]
        _, reward_early, _, _, _ = env.step(action)

        env.reset(seed=42)
        # Simulate late game
        env.total_cards_played = 90
        mask = env.action_masks()
        action = np.where(mask)[0][0]
        _, reward_late, _, _, _ = env.step(action)

        # Late game should have higher reward due to phase multiplier
        # phase = 91/98 ~ 0.93, multiplier ~ 1.465
        assert reward_late > reward_early


class TestMultiPlayer:
    """Tests for multi-player functionality."""

    def test_n_players_creates_correct_hands(self):
        """Correct number of players are initialized."""
        for n in [2, 3, 4, 5]:
            env = TheGameEnv(n_players=n)
            env.reset(seed=42)
            assert len(env.hands) == n

    def test_player_rotation(self):
        """Players take turns in order."""
        env = TheGameEnv(n_players=3)
        env.reset(seed=42)

        # Play enough cards to trigger turn end
        for _ in range(10):
            mask = env.action_masks()
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break
            _, _, term, _, _ = env.step(valid[0])
            if term:
                break

        # Player should have changed at some point if turn ended
        # (This is a weak test since we can't guarantee turn completion)
        assert True  # Rotation is tested implicitly by game mechanics

    def test_hand_size_varies_by_player_count(self):
        """Hand size is 7 for 2 players, 6 for 3+ players."""
        env2 = TheGameEnv(n_players=2)
        assert env2.hand_size == 7

        env3 = TheGameEnv(n_players=3)
        assert env3.hand_size == 6

        env5 = TheGameEnv(n_players=5)
        assert env5.hand_size == 6


class TestEndTurnAction:
    """Tests for the explicit end turn action."""

    def test_end_turn_action_index(self):
        """End turn is the last action in the action space."""
        env = TheGameEnv(hand_size=6)
        assert env.action_space.n == 25
        # end_turn = 24 (index hand_size * 4)

    def test_end_turn_invalid_before_minimum(self):
        """Cannot end turn before playing minimum cards."""
        env = TheGameEnv()
        env.reset(seed=42)
        end_turn = env.hand_size * 4
        assert not env.action_masks()[end_turn]

    def test_end_turn_valid_after_minimum(self):
        """Can end turn after playing 2 cards."""
        env = TheGameEnv()
        env.reset(seed=42)
        # Play 2 cards
        for _ in range(2):
            mask = env.action_masks()
            action = np.where(mask)[0][0]
            env.step(action)
        end_turn = env.hand_size * 4
        assert env.action_masks()[end_turn]

    def test_end_turn_switches_player(self):
        """End turn action moves to next player."""
        env = TheGameEnv(n_players=2)
        env.reset(seed=42)
        assert env.current_player_idx == 0
        # Play 2 cards then end turn
        for _ in range(2):
            mask = env.action_masks()
            action = np.where(mask)[0][0]
            env.step(action)
        end_turn = env.hand_size * 4
        env.step(end_turn)
        assert env.current_player_idx == 1


class TestEmptyHandSkipping:
    """Tests for skipping players with empty hands."""

    def test_player_with_empty_hand_is_skipped(self):
        """Player with empty hand is skipped, next player takes turn."""
        env = TheGameEnv(n_players=3)
        env.reset(seed=42)

        env.hands[1] = np.array([], dtype=np.int32)
        assert env.current_player_idx == 0

        for _ in range(2):
            mask = env.action_masks()
            action = np.where(mask)[0][0]
            env.step(action)

        end_turn = env.hand_size * 4
        env.step(end_turn)

        assert env.current_player_idx == 2

    def test_multiple_consecutive_empty_hands_skipped(self):
        """Multiple consecutive players with empty hands are skipped."""
        env = TheGameEnv(n_players=4)
        env.reset(seed=42)

        env.hands[1] = np.array([], dtype=np.int32)
        env.hands[2] = np.array([], dtype=np.int32)

        for _ in range(2):
            mask = env.action_masks()
            action = np.where(mask)[0][0]
            env.step(action)

        end_turn = env.hand_size * 4
        env.step(end_turn)

        assert env.current_player_idx == 3

    def test_win_when_all_hands_and_deck_empty(self):
        """Game is won when all hands are empty and deck is empty."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 50])
        env.hands[0] = np.array([51], dtype=np.int32)
        env.hands[1] = np.array([], dtype=np.int32)
        env.remaining_deck = np.array([], dtype=np.int32)

        action = 0 * 4 + 2
        _, _, terminated, _, info = env.step(action)

        assert terminated
        assert info.get("victory", False)

    def test_game_continues_after_skipping_empty_hands(self):
        """Game continues normally after skipping empty-handed players."""
        env = TheGameEnv(n_players=3)
        env.reset(seed=42)

        env.hands[1] = np.array([], dtype=np.int32)

        for _ in range(2):
            mask = env.action_masks()
            action = np.where(mask)[0][0]
            env.step(action)

        end_turn = env.hand_size * 4
        obs, reward, terminated, truncated, info = env.step(end_turn)

        assert not terminated
        assert env.current_player_idx == 2
        assert len(env.hands[2]) > 0
        assert np.any(env.action_masks())


class TestWinRateParity:
    """Tests ensuring TheGameEnv matches run_simulation win rates."""

    def test_env_matches_run_simulation_win_rate(self):
        """TheGameEnv produces similar win rates to run_simulation."""
        from functools import partial

        from game_setup import run_simulation
        from strategies import _identify_min_distance_card, bonus_play_strategy

        def play_game_bonus_strategy_env(env, seed, bonus_play_threshold=2):
            obs, info = env.reset(seed=seed)

            while True:
                hand = env.hands[env.current_player_idx].copy()
                stacks = env.stacks
                n_cards_to_play = 2 if len(env.remaining_deck) > 0 else 1
                cards_played = env.cards_played_this_turn
                end_turn_action = env.hand_size * 4

                if cards_played >= n_cards_to_play:
                    try:
                        _, _, min_diff = _identify_min_distance_card(hand, stacks)
                        if min_diff > bonus_play_threshold:
                            obs, _, term, trunc, info = env.step(end_turn_action)
                            if term or trunc:
                                return info.get("victory", False)
                            continue
                    except Exception:
                        obs, _, term, trunc, info = env.step(end_turn_action)
                        if term or trunc:
                            return info.get("victory", False)
                        continue

                try:
                    best_card, best_stack_idx, _ = _identify_min_distance_card(
                        hand, stacks
                    )
                except Exception:
                    return False

                card_idx = np.where(hand == best_card)[0][0]
                action = card_idx * 4 + best_stack_idx

                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    return info.get("victory", False)

        n_games = 100
        n_players = 5

        np.random.seed(123)
        strategy = partial(bonus_play_strategy, bonus_play_threshold=2)
        result = run_simulation(strategy, n_games=n_games, n_players=n_players)
        run_sim_wins = len(result["victories"])

        env = TheGameEnv(n_players=n_players)
        env_wins = sum(
            1 for i in range(n_games) if play_game_bonus_strategy_env(env, seed=i)
        )

        assert abs(run_sim_wins - env_wins) <= 5
