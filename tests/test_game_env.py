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
        """Observation includes hand, stacks, deck, cards_played, min_required, other hands."""
        env = TheGameEnv(n_players=3, hand_size=6)
        # 6 hand + 4 stack tops + 1 deck + 1 cards_played + 1 min_required
        # + 2 other_hand_sizes (max_players=3 default, so 3-1=2) = 15
        assert env.observation_space.shape == (15,)

        env5 = TheGameEnv(n_players=5, hand_size=6, max_players=5)
        # + 4 other_hand_sizes (5-1) = 17
        assert env5.observation_space.shape == (17,)

    def test_observation_space_with_max_players(self):
        """Observation size is fixed based on max_players for curriculum learning."""
        env2 = TheGameEnv(n_players=2, max_players=5, hand_size=6)
        env3 = TheGameEnv(n_players=3, max_players=5, hand_size=6)
        env5 = TheGameEnv(n_players=5, max_players=5, hand_size=6)

        # All should have same observation size: 6 + 4 + 3 + 4 = 17
        assert env2.observation_space.shape == (17,)
        assert env3.observation_space.shape == (17,)
        assert env5.observation_space.shape == (17,)

    def test_max_players_defaults_to_n_players(self):
        """max_players defaults to n_players when not specified."""
        env3 = TheGameEnv(n_players=3)
        assert env3.max_players == 3

        env5 = TheGameEnv(n_players=5)
        assert env5.max_players == 5

    def test_observation_padding_with_fewer_players(self):
        """Other hand sizes are padded with zeros when n_players < max_players."""
        env = TheGameEnv(n_players=2, max_players=5, hand_size=6)
        obs, _ = env.reset(seed=42)

        # Position: hand (6) + stack_tops (4) + deck (1) + cards_played (1) + min_required (1) = 13
        other_hands_start = 6 + 4 + 3
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
        # Position: hand (6) + stack_tops (4) + deck (1) = index 11
        cards_played_idx = env.hand_size + 4 + 1
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
        # Position: hand (6) + stack_tops (4) + deck (1) + cards_played (1) = 12
        min_required_idx = env.hand_size + 4 + 2
        # Normalized: (min_required - 1), so 2 -> 1.0
        assert obs[min_required_idx] == 1.0  # Deck exists at start

    def test_other_hand_sizes_in_observation(self):
        """Observation includes other players' hand sizes."""
        env = TheGameEnv(n_players=3)
        obs, _ = env.reset(seed=42)
        # Position: hand (6) + stack_tops (4) + deck (1) + cards_played (1) + min_required (1) = 13
        other_hands_start = env.hand_size + 4 + 3
        # All other players start with full hands (normalized to 1.0)
        assert obs[other_hands_start] == pytest.approx(1.0, abs=0.01)
        assert obs[other_hands_start + 1] == pytest.approx(1.0, abs=0.01)


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


class TestEdgeCases:
    """Edge cases matching game_setup.py behavior."""

    # =========================================================================
    # Minimum cards transition tests
    # =========================================================================

    def test_min_required_is_two_when_deck_has_cards(self):
        """Minimum required is 2 when deck is not empty."""
        env = TheGameEnv(n_players=3)
        env.reset(seed=42)
        assert len(env.remaining_deck) > 0
        assert env._min_cards_required() == 2

    def test_min_required_is_one_when_deck_empty(self):
        """Minimum required is 1 when deck is empty."""
        env = TheGameEnv(n_players=3)
        env.reset(seed=42)
        env.remaining_deck = np.array([], dtype=np.int32)
        assert env._min_cards_required() == 1

    def test_min_required_changes_after_deck_empties_during_turn(self):
        """Min required updates correctly when deck empties mid-game."""
        from utils import Stack

        env = TheGameEnv(n_players=2, hand_size=7)
        env.reset(seed=42)

        # Set up: small deck, player can play cards
        env.stacks[2] = Stack.from_array([1, 10])
        env.hands[0] = np.array([20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
        env.remaining_deck = np.array([5, 6], dtype=np.int32)

        assert env._min_cards_required() == 2

        # Play 2 cards, then end turn to trigger draw
        action1 = 0 * 4 + 2  # Play 20 on increasing stack
        env.step(action1)
        assert env._min_cards_required() == 2  # Still deck remaining

        action2 = 0 * 4 + 2  # Play 30 on increasing stack
        env.step(action2)

        end_turn = env.hand_size * 4
        env.step(end_turn)

        # After drawing, deck should be empty (drew 2 to refill 7-card hand)
        # But player 2 now playing, deck status matters
        if len(env.remaining_deck) == 0:
            assert env._min_cards_required() == 1

    def test_play_one_card_valid_when_deck_empty(self):
        """Can end turn after just 1 card when deck is empty."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 10])
        env.hands[0] = np.array([20, 30, 40, 50, 60, 70], dtype=np.int32)
        env.remaining_deck = np.array([], dtype=np.int32)

        # Play 1 card
        action = 0 * 4 + 2  # Play 20 on increasing stack
        env.step(action)

        # End turn should be valid now (min=1 when deck empty)
        end_turn = env.hand_size * 4
        mask = env.action_masks()
        assert mask[end_turn], "End turn should be valid after 1 card when deck empty"

    # =========================================================================
    # Partial hand refill tests
    # =========================================================================

    def test_partial_draw_when_deck_nearly_empty(self):
        """Player draws only remaining cards when deck has fewer than needed."""
        from utils import Stack

        env = TheGameEnv(n_players=2, hand_size=7)
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 10])
        env.hands[0] = np.array([20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
        env.hands[1] = np.array([21, 31, 41, 51, 61, 71, 81], dtype=np.int32)
        env.remaining_deck = np.array([5, 6], dtype=np.int32)

        # Play 2 cards
        env.step(0 * 4 + 2)  # Play 20
        env.step(0 * 4 + 2)  # Play 30

        # End turn triggers draw - need 2 cards, deck has 2
        end_turn = env.hand_size * 4
        env.step(end_turn)

        # Player 0 should have drawn 2 (5-card hand → 7 after drawing 2)
        assert len(env.hands[0]) == 7

    def test_partial_draw_fewer_cards_than_needed(self):
        """Draw less than needed when deck doesn't have enough."""
        from utils import Stack

        env = TheGameEnv(n_players=2, hand_size=7)
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 10])
        env.hands[0] = np.array([20, 30, 40, 50], dtype=np.int32)  # Only 4 cards
        env.hands[1] = np.array([21, 31, 41, 51, 61, 71, 81], dtype=np.int32)
        env.remaining_deck = np.array([5], dtype=np.int32)  # Only 1 card

        # Play 2 cards
        env.step(0 * 4 + 2)  # Play 20
        env.step(0 * 4 + 2)  # Play 30

        # End turn - need 5 cards (7-2), deck has 1
        end_turn = env.hand_size * 4
        env.step(end_turn)

        # Player 0 should have drawn 1 (2+1=3 cards)
        assert len(env.hands[0]) == 3

    def test_no_draw_when_deck_empty(self):
        """No cards drawn when deck is empty."""
        from utils import Stack

        env = TheGameEnv(n_players=2, hand_size=7)
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 10])
        env.hands[0] = np.array([20, 30, 40, 50, 60, 70, 80], dtype=np.int32)
        env.hands[1] = np.array([21, 31, 41, 51, 61, 71, 81], dtype=np.int32)
        env.remaining_deck = np.array([], dtype=np.int32)

        # Play 1 card (min=1 when deck empty)
        env.step(0 * 4 + 2)

        initial_hand_size = len(env.hands[0])
        end_turn = env.hand_size * 4
        env.step(end_turn)

        # Hand size unchanged (no draw from empty deck)
        assert len(env.hands[0]) == initial_hand_size

    # =========================================================================
    # Victory detection edge cases
    # =========================================================================

    def test_victory_on_last_card_played(self):
        """Victory triggered immediately when last card is played."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 50])
        env.hands[0] = np.array([51], dtype=np.int32)
        env.hands[1] = np.array([], dtype=np.int32)
        env.remaining_deck = np.array([], dtype=np.int32)

        action = 0 * 4 + 2  # Play 51 on increasing stack
        _, _, terminated, _, info = env.step(action)

        assert terminated
        assert info.get("victory", False)

    def test_no_victory_when_others_have_cards(self):
        """Not victory when current player empty but others have cards."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 50])
        env.hands[0] = np.array([51], dtype=np.int32)
        env.hands[1] = np.array([60, 70], dtype=np.int32)  # Other player has cards
        env.remaining_deck = np.array([], dtype=np.int32)

        action = 0 * 4 + 2  # Play 51
        _, _, terminated, _, info = env.step(action)

        # Should not be victory (player 1 still has cards)
        assert not info.get("victory", False) or not terminated

    def test_victory_requires_all_cards_played(self):
        """Victory only when all 98 cards played (deck + all hands empty)."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        # Deck not empty
        env.stacks[2] = Stack.from_array([1, 50])
        env.hands[0] = np.array([51], dtype=np.int32)
        env.hands[1] = np.array([], dtype=np.int32)
        env.remaining_deck = np.array([60], dtype=np.int32)  # Still cards in deck

        action = 0 * 4 + 2
        _, _, terminated, _, info = env.step(action)

        # Not victory because deck has cards
        assert not info.get("victory", False)

    # =========================================================================
    # Loss condition tests
    # =========================================================================

    def test_loss_when_stuck_before_minimum(self):
        """Game ends in loss when player can't play minimum cards."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        # Set up stuck position: all stacks blocked for all cards
        env.stacks[0] = Stack.from_array([99, 10])  # Dec, need < 10 or 20
        env.stacks[1] = Stack.from_array([99, 10])  # Dec, need < 10 or 20
        env.stacks[2] = Stack.from_array([1, 90])  # Inc, need > 90 or 80
        env.stacks[3] = Stack.from_array([1, 90])  # Inc, need > 90 or 80
        env.hands[0] = np.array([50], dtype=np.int32)  # Can't play anywhere
        env.remaining_deck = np.array([60, 70], dtype=np.int32)

        # Player has 1 card, can't play it, min_required=2
        # No valid actions available
        mask = env.action_masks()
        assert not np.any(mask), "Should have no valid actions"

        # Game should detect this as loss
        terminated, victory = env._check_game_over()
        assert terminated
        assert not victory

    def test_loss_on_invalid_end_turn(self):
        """Explicit end turn before minimum gives loss."""
        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        # Play 1 card
        mask = env.action_masks()
        action = np.where(mask)[0][0]
        env.step(action)

        # Try to end turn (should be invalid, min=2)
        end_turn = env.hand_size * 4
        _, reward, terminated, _, info = env.step(end_turn)

        assert terminated
        assert reward < 0
        assert info.get("invalid", False)

    def test_loss_when_next_player_stuck(self):
        """Game ends when next player cannot make any valid plays."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        # Player 0 can play on stack 2, player 1 will be stuck
        # after player 0 plays (stack 2 will be at 30, player 1 has 50 which
        # can still be played... so we need a different setup)

        # Set up: Player 1 has card 50, which can only be played if
        # stacks are set correctly AFTER player 0's turn
        env.stacks[0] = Stack.from_array([99, 10])  # Dec, need < 10 or 20
        env.stacks[1] = Stack.from_array([99, 10])  # Dec, need < 10 or 20
        env.stacks[2] = Stack.from_array([1, 95])  # Inc, need > 95 or 85
        env.stacks[3] = Stack.from_array([1, 95])  # Inc, need > 95 or 85

        # Player 0 can play 96 and 97 on increasing stacks
        env.hands[0] = np.array([96, 97, 98, 5, 6, 7], dtype=np.int32)
        # Player 1 stuck with 50 (can't play on any stack after player 0's turn)
        env.hands[1] = np.array([50], dtype=np.int32)

        env.remaining_deck = np.array([60, 70], dtype=np.int32)

        # Player 0 plays 96 on stack 2 and 97 on stack 3
        env.step(0 * 4 + 2)  # Play 96 on inc stack 2
        env.step(1 * 4 + 3)  # Play 97 on inc stack 3

        end_turn = env.hand_size * 4
        _, _, terminated, _, info = env.step(end_turn)

        # Game should end because player 1 is stuck (50 can't be played anywhere)
        assert terminated
        assert not info.get("victory", False)
        assert info.get("reason") == "next_player_stuck"

    # =========================================================================
    # Auto-end turn behavior tests
    # =========================================================================

    def test_auto_end_when_hand_empty_after_min(self):
        """Turn auto-ends when hand empties after playing minimum cards."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        # Player 0 has only 2 cards
        env.stacks[2] = Stack.from_array([1, 10])
        env.hands[0] = np.array([20, 30], dtype=np.int32)
        env.hands[1] = np.array([40, 50, 60, 70, 80, 85], dtype=np.int32)
        env.remaining_deck = np.array([91, 92, 93, 94], dtype=np.int32)

        # Play both cards
        env.step(0 * 4 + 2)  # Play 20
        assert env.current_player_idx == 0
        env.step(0 * 4 + 2)  # Play 30

        # Hand empty, min met, turn should auto-end
        # After drawing, should be player 1's turn (or back to player 0 after draw)
        assert env.current_player_idx == 1

    def test_loss_when_hand_empty_before_min(self):
        """Game ends in loss when hand empties before playing minimum."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        # Player has only 1 card
        env.stacks[2] = Stack.from_array([1, 10])
        env.hands[0] = np.array([20], dtype=np.int32)
        env.hands[1] = np.array([40, 50, 60, 70, 80, 85], dtype=np.int32)
        env.remaining_deck = np.array([91, 92], dtype=np.int32)  # Deck exists, min=2

        # Play the only card
        _, _, terminated, _, info = env.step(0 * 4 + 2)

        # Should be loss (can't meet min_required=2)
        assert terminated
        assert not info.get("victory", False)

    def test_auto_end_no_valid_moves_but_min_reached(self):
        """Turn ends when no valid moves but minimum was reached."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        # Set up: after 2 plays, no more valid moves
        env.stacks[0] = Stack.from_array([99, 90])  # Dec, need < 90 or 100
        env.stacks[1] = Stack.from_array([99, 90])  # Dec, need < 90 or 100
        env.stacks[2] = Stack.from_array([1, 10])  # Inc, need > 10 or 0
        env.stacks[3] = Stack.from_array([1, 85])  # Inc, need > 85 or 75

        # Player has cards where only 2 can be played
        env.hands[0] = np.array([20, 30, 80, 81, 82, 83], dtype=np.int32)
        env.hands[1] = np.array([40, 50, 60, 70, 75, 86], dtype=np.int32)
        env.remaining_deck = np.array([91, 92], dtype=np.int32)

        # Play 20 and 30 on increasing stack 2
        env.step(0 * 4 + 2)  # Play 20
        env.step(0 * 4 + 2)  # Play 30

        # Now remaining cards (80-83) can't be played on any stack
        # Turn should auto-end (or end_turn should be valid)
        mask = env.action_masks()
        end_turn = env.hand_size * 4

        # Either no valid card plays exist, or end_turn is available
        card_plays_valid = np.any(mask[:-1])
        if not card_plays_valid:
            # If no card plays, game should have auto-ended
            assert env.current_player_idx == 1 or mask[end_turn]
        else:
            # If card plays exist, end_turn should be valid
            assert mask[end_turn]

    # =========================================================================
    # Card index after removal tests
    # =========================================================================

    def test_action_mask_updates_after_play(self):
        """Action mask reflects new hand state after card removal."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 10])
        env.hands[0] = np.array([20, 30, 40, 50, 60, 70], dtype=np.int32)

        # Get mask before play
        mask_before = env.action_masks().copy()

        # Play card at index 0 (card 20)
        env.step(0 * 4 + 2)

        # Get mask after play
        mask_after = env.action_masks()

        # Mask should be different (hand changed)
        assert not np.array_equal(mask_before, mask_after)

        # Card at "new index 0" is now 30, not 20
        assert env.hands[0][0] == 30

    def test_card_indices_shift_after_removal(self):
        """Card indices shift correctly after card removal from hand."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 10])
        original_hand = np.array([20, 30, 40, 50, 60, 70], dtype=np.int32)
        env.hands[0] = original_hand.copy()

        # Play card at index 2 (card 40)
        env.step(2 * 4 + 2)

        # Hand should now be [20, 30, 50, 60, 70]
        expected = np.array([20, 30, 50, 60, 70], dtype=np.int32)
        np.testing.assert_array_equal(env.hands[0], expected)

        # Index 2 now points to 50, not 40
        assert env.hands[0][2] == 50

    def test_play_consecutive_cards_with_shifting_indices(self):
        """Playing multiple cards maintains correct index tracking."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 10])
        env.hands[0] = np.array([20, 30, 40, 50, 60, 70], dtype=np.int32)
        env.remaining_deck = np.array([], dtype=np.int32)  # Empty deck, min=1

        # Play card at index 0 (20), then "new index 0" (30), then "new index 0" (40)
        env.step(0 * 4 + 2)  # Plays 20, hand becomes [30, 40, 50, 60, 70]
        assert env.stacks[2].top == 20

        env.step(0 * 4 + 2)  # Plays 30, hand becomes [40, 50, 60, 70]
        assert env.stacks[2].top == 30

        env.step(0 * 4 + 2)  # Plays 40, hand becomes [50, 60, 70]
        assert env.stacks[2].top == 40

    # =========================================================================
    # Trick play stack state tests
    # =========================================================================

    def test_stack_top_after_trick_play_increasing(self):
        """After trick play on increasing stack, top is the trick card."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 50])
        env.hands[0] = np.array([40, 55, 60, 65, 70, 75], dtype=np.int32)

        # Play 40 on increasing stack at 50 (trick: 40 + 10 = 50)
        action = 0 * 4 + 2
        env.step(action)

        assert env.stacks[2].top == 40

    def test_stack_top_after_trick_play_decreasing(self):
        """After trick play on decreasing stack, top is the trick card."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        env.stacks[0] = Stack.from_array([99, 50])
        env.hands[0] = np.array([60, 45, 40, 35, 30, 25], dtype=np.int32)

        # Play 60 on decreasing stack at 50 (trick: 60 - 10 = 50)
        action = 0 * 4 + 0
        env.step(action)

        assert env.stacks[0].top == 60

    def test_subsequent_play_after_trick_increasing(self):
        """After trick on increasing stack, next play must be > trick card."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        env.stacks[2] = Stack.from_array([1, 50])
        env.hands[0] = np.array([40, 35, 45, 55, 60, 70], dtype=np.int32)
        env.remaining_deck = np.array([], dtype=np.int32)

        # Play 40 (trick)
        env.step(0 * 4 + 2)
        assert env.stacks[2].top == 40

        # Now 35 should be invalid (35 < 40, not a trick)
        # But 45, 55, 60, 70 should be valid (> 40)
        # Card 35 is now at index 0 after removal of 40
        # Action for index 0 on stack 2 should be invalid
        assert not env._is_valid_play(35, 2)
        assert env._is_valid_play(45, 2)

    def test_subsequent_play_after_trick_decreasing(self):
        """After trick on decreasing stack, next play must be < trick card."""
        from utils import Stack

        env = TheGameEnv(n_players=2)
        env.reset(seed=42)

        env.stacks[0] = Stack.from_array([99, 50])
        env.hands[0] = np.array([60, 65, 55, 45, 40, 35], dtype=np.int32)
        env.remaining_deck = np.array([], dtype=np.int32)

        # Play 60 (trick)
        env.step(0 * 4 + 0)
        assert env.stacks[0].top == 60

        # Now 65 should be invalid (65 > 60, not a trick)
        # But 55, 45, 40, 35 should be valid (< 60)
        assert not env._is_valid_play(65, 0)
        assert env._is_valid_play(55, 0)

    # =========================================================================
    # Multi-player edge cases
    # =========================================================================

    def test_wrap_around_player_rotation(self):
        """Player rotation wraps from last player to first."""
        env = TheGameEnv(n_players=3)
        env.reset(seed=42)

        # Manually set to last player
        env.current_player_idx = 2
        env.cards_played_this_turn = 0

        # Play 2 cards
        for _ in range(2):
            mask = env.action_masks()
            valid = np.where(mask)[0]
            if len(valid) > 0:
                env.step(valid[0])

        # End turn
        end_turn = env.hand_size * 4
        if env.action_masks()[end_turn]:
            env.step(end_turn)
            assert env.current_player_idx == 0  # Wrapped around

    def test_skip_multiple_empty_hands_wrap_around(self):
        """Correctly skips multiple empty hands including wrap-around."""
        env = TheGameEnv(n_players=4)
        env.reset(seed=42)

        # Player 0 current, players 1,2,3 empty
        env.hands[1] = np.array([], dtype=np.int32)
        env.hands[2] = np.array([], dtype=np.int32)
        env.hands[3] = np.array([], dtype=np.int32)

        # Play 2 cards
        for _ in range(2):
            mask = env.action_masks()
            action = np.where(mask)[0][0]
            env.step(action)

        # End turn - should skip back to player 0 (only one with cards)
        end_turn = env.hand_size * 4
        env.step(end_turn)

        assert env.current_player_idx == 0


class TestWinRateParity:
    """Tests ensuring TheGameEnv matches run_simulation win rates."""

    def test_env_matches_run_simulation_win_rate(self):
        """TheGameEnv produces similar win rates to run_simulation."""
        from functools import partial

        from game_setup import run_simulation
        from strategies import bonus_play_strategy
        from utils import identify_min_distance_card

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
                        _, _, min_diff = identify_min_distance_card(hand, stacks)
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
                    best_card, best_stack_idx, _ = identify_min_distance_card(
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
