"""Gymnasium environment for The Game (multi-player cooperative card game).

Training perspective: single player's turn-by-turn decisions.
All players share the same policy (centralized training with parameter sharing).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils import Stack, play_to_stack


class TheGameEnv(gym.Env):
    """Multi-player cooperative RL environment for The Game.

    The agent controls all players, but observes from the current player's
    perspective. Each step is a single card play. A turn consists of playing
    at least 2 cards (or 1 if deck is empty), then drawing back to hand size.

    Args:
        n_players: Number of players in the game.
        max_players: Maximum players for fixed observation size (for curriculum learning).
        hand_size: Cards per player (6 for 3+ players, 7 for 2 players).
        reward_per_card: Reward for each successfully played card.
        win_reward: Bonus reward for winning the game.
        loss_penalty: Penalty for losing the game.
        trick_play_reward: Bonus for backwards trick plays (±10 reset cards).
        distance_penalty_scale: Scale for distance-based reward shaping.
        progress_reward_scale: Scale for progress bonus on loss.
        stack_health_scale: Scale for balanced stack usage reward.
        phase_multiplier_scale: Scale for late-game reward amplification.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_players=3,
        max_players=None,
        hand_size=None,
        reward_per_card=0.01,
        win_reward=1.0,
        loss_penalty=0.5,
        trick_play_reward=0.1,
        distance_penalty_scale=0.001,
        progress_reward_scale=0.0,
        stack_health_scale=0.01,
        phase_multiplier_scale=0.5,
    ):
        super().__init__()
        self.n_players = n_players
        self.max_players = max_players if max_players is not None else n_players
        self.hand_size = hand_size if hand_size else (6 if n_players > 2 else 7)
        self.reward_per_card = reward_per_card
        self.win_reward = win_reward
        self.loss_penalty = loss_penalty
        self.trick_play_reward = trick_play_reward
        self.distance_penalty_scale = distance_penalty_scale
        self.progress_reward_scale = progress_reward_scale
        self.stack_health_scale = stack_health_scale
        self.phase_multiplier_scale = phase_multiplier_scale

        # Action: card_index * 4 + stack_index, plus one "end turn" action
        # card_index in [0, hand_size-1], stack_index in [0, 3]
        # Last action (hand_size * 4) = end turn
        self.action_space = spaces.Discrete(self.hand_size * 4 + 1)

        # Observation: [hand (hand_size), stack_tops (4),
        #               deck_remaining (1), cards_played_this_turn (1), min_cards_required (1),
        #               other_hand_sizes (max_players - 1)]
        # All values normalized to [0, 1] range
        obs_size = self.hand_size + 4 + 3 + (self.max_players - 1)
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self._reset_game_state()

    def _reset_game_state(self):
        """Initialize internal game state."""
        self.hands = []
        self.stacks = []
        self.remaining_deck = np.array([], dtype=np.int32)
        self.current_player_idx = 0
        self.cards_played_this_turn = 0
        self.total_cards_played = 0
        self.total_turns = 0
        self.total_distance = 0

    def _shuffle_and_deal(self, seed=None):
        """Shuffle deck and deal cards to all players."""
        if seed is not None:
            np.random.seed(seed)

        deck = np.arange(2, 100, dtype=np.int32)
        np.random.shuffle(deck)

        self.hands = []
        card_idx = 0
        for _ in range(self.n_players):
            self.hands.append(deck[card_idx : card_idx + self.hand_size])
            card_idx += self.hand_size

        self.remaining_deck = deck[card_idx:]

        self.stacks = [
            Stack(99),  # decreasing_1
            Stack(99),  # decreasing_2
            Stack(1),  # increasing_1
            Stack(1),  # increasing_2
        ]

    def _get_observation(self):
        """Build observation array from current player's perspective.

        All values normalized to [0, 1] range for better learning.
        """
        hand = self.hands[self.current_player_idx]

        # Pad hand to fixed size and normalize (0 = empty slot, cards 2-99)
        hand_obs = np.zeros(self.hand_size, dtype=np.float32)
        hand_obs[: len(hand)] = hand / 100.0  # Normalize to [0, 1]

        # Stack tops normalized
        stack_tops = np.array([s.top / 100.0 for s in self.stacks], dtype=np.float32)

        # Deck size normalized (max 98 - n_players * hand_size)
        max_deck = 98 - self.n_players * self.hand_size
        deck_remaining = np.array(
            [min(len(self.remaining_deck) / max(max_deck, 1), 1.0)], dtype=np.float32
        )

        # Cards played this turn normalized (max is hand_size)
        cards_played = np.array(
            [self.cards_played_this_turn / self.hand_size], dtype=np.float32
        )

        # Min required normalized (1 or 2)
        min_required = np.array(
            [(self._min_cards_required() - 1)], dtype=np.float32
        )  # 0 or 1

        # Other players' hand sizes (normalized by hand_size)
        other_hand_sizes = np.zeros(self.max_players - 1, dtype=np.float32)
        for i in range(1, min(self.n_players, self.max_players)):
            player_idx = (self.current_player_idx + i) % self.n_players
            other_hand_sizes[i - 1] = len(self.hands[player_idx]) / self.hand_size

        return np.concatenate(
            [
                hand_obs,
                stack_tops,
                deck_remaining,
                cards_played,
                min_required,
                other_hand_sizes,
            ]
        )

    def _min_cards_required(self):
        """Minimum cards to play this turn (2 if deck exists, 1 if empty)."""
        return 2 if len(self.remaining_deck) > 0 else 1

    def _is_valid_play(self, card, stack_idx):
        """Check if playing card on stack is valid."""
        if card == 0:  # Empty slot in hand
            return False

        stack = self.stacks[stack_idx]
        top_card = stack.top

        if stack_idx >= 2:  # Increasing stacks
            return card > top_card or card + 10 == top_card
        else:  # Decreasing stacks
            return card < top_card or card - 10 == top_card

    def action_masks(self):
        """Return boolean mask of valid actions for current player.

        Returns:
            np.ndarray of shape (hand_size * 4 + 1,) with True for valid actions.
            The last index is the "end turn" action.
        """
        hand = self.hands[self.current_player_idx]
        mask = np.zeros(self.hand_size * 4 + 1, dtype=bool)

        for card_idx in range(self.hand_size):
            if card_idx >= len(hand):
                continue  # Empty slot
            card = hand[card_idx]
            for stack_idx in range(4):
                if self._is_valid_play(card, stack_idx):
                    action = card_idx * 4 + stack_idx
                    mask[action] = True

        # End turn action is valid if minimum cards have been played
        end_turn_action = self.hand_size * 4
        if self.cards_played_this_turn >= self._min_cards_required():
            mask[end_turn_action] = True

        return mask

    def _decode_action(self, action):
        """Convert action integer to (card_index, stack_index)."""
        card_idx = action // 4
        stack_idx = action % 4
        return card_idx, stack_idx

    def _end_turn(self):
        """End current turn: draw cards, move to next player.

        Skips players with empty hands, matching the behavior of run_game()
        in game_setup.py.
        """
        hand = self.hands[self.current_player_idx]
        cards_to_draw = self.hand_size - len(hand)

        if cards_to_draw > 0 and len(self.remaining_deck) > 0:
            draw_count = min(cards_to_draw, len(self.remaining_deck))
            drawn = self.remaining_deck[:draw_count]
            self.hands[self.current_player_idx] = np.concatenate([hand, drawn])
            self.remaining_deck = self.remaining_deck[draw_count:]

        for _ in range(self.n_players):
            self.current_player_idx = (self.current_player_idx + 1) % self.n_players
            if len(self.hands[self.current_player_idx]) > 0:
                break

        self.cards_played_this_turn = 0
        self.total_turns += 1

    def _get_episode_stats(self):
        """Return statistics for the completed episode."""
        avg_cards_per_turn = (
            self.total_cards_played / self.total_turns if self.total_turns > 0 else 0
        )
        avg_distance = (
            self.total_distance / self.total_cards_played
            if self.total_cards_played > 0
            else 0
        )
        return {
            "total_cards_played": self.total_cards_played,
            "total_turns": self.total_turns,
            "avg_cards_per_turn": avg_cards_per_turn,
            "avg_distance": avg_distance,
        }

    def _check_game_over(self):
        """Check if game is won or lost."""
        total_cards = len(self.remaining_deck) + sum(len(h) for h in self.hands)

        if total_cards == 0:
            return True, True  # terminated, victory

        # Check if any valid move exists for current player
        if not np.any(self.action_masks()):
            if self.cards_played_this_turn < self._min_cards_required():
                return True, False  # terminated, loss

        return False, False

    def reset(self, seed=None, options=None):
        """Reset the environment to a new game.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            observation: Initial observation.
            info: Additional information.
        """
        super().reset(seed=seed)
        self._shuffle_and_deal(seed)
        self.current_player_idx = 0
        self.cards_played_this_turn = 0
        self.total_cards_played = 0
        self.total_turns = 0
        self.total_distance = 0

        return self._get_observation(), {"action_mask": self.action_masks()}

    def step(self, action):
        """Execute one card play action or end turn.

        Args:
            action: Integer in [0, hand_size * 4] encoding (card_idx, stack_idx)
                or end turn (hand_size * 4).

        Returns:
            observation: New game state.
            reward: Reward for this action.
            terminated: Whether game ended (win or loss).
            truncated: Always False (no time limit).
            info: Additional information including action mask.
        """
        end_turn_action = self.hand_size * 4

        # Handle end turn action
        if action == end_turn_action:
            if self.cards_played_this_turn < self._min_cards_required():
                info = {"invalid": True, **self._get_episode_stats()}
                return self._get_observation(), -1.0, True, False, info

            self._end_turn()

            # Check if next player can play
            if not np.any(self.action_masks()):
                info = {"victory": False, "reason": "next_player_stuck"}
                info.update(self._get_episode_stats())
                progress_bonus = self.progress_reward_scale * (
                    self.total_cards_played / 98
                )
                return (
                    self._get_observation(),
                    -self.loss_penalty + progress_bonus,
                    True,
                    False,
                    info,
                )

            return (
                self._get_observation(),
                0.0,
                False,
                False,
                {"action_mask": self.action_masks()},
            )

        card_idx, stack_idx = self._decode_action(action)
        hand = self.hands[self.current_player_idx]

        # Validate action
        if card_idx >= len(hand):
            info = {"invalid": True, **self._get_episode_stats()}
            return self._get_observation(), -1.0, True, False, info

        card = hand[card_idx]

        if not self._is_valid_play(card, stack_idx):
            info = {"invalid": True, **self._get_episode_stats()}
            return self._get_observation(), -1.0, True, False, info

        # Execute the play using existing utility
        try:
            stack_top = self.stacks[stack_idx].top
            distance = abs(card - stack_top)
            new_hand, new_stacks = play_to_stack(hand, card, stack_idx, self.stacks)
            self.hands[self.current_player_idx] = new_hand
            self.stacks = new_stacks
            self.cards_played_this_turn += 1
            self.total_cards_played += 1
            self.total_distance += distance
        except ValueError:
            info = {"invalid": True, **self._get_episode_stats()}
            return self._get_observation(), -1.0, True, False, info

        reward = self.reward_per_card

        # Bonus for BACKWARDS trick play (reset card going against stack direction)
        # Increasing stack (idx >= 2): trick = card + 10 == stack_top (card is 10 BELOW)
        # Decreasing stack (idx < 2): trick = card - 10 == stack_top (card is 10 ABOVE)
        is_trick_play = (stack_idx >= 2 and card + 10 == stack_top) or (
            stack_idx < 2 and card - 10 == stack_top
        )
        if is_trick_play:
            reward += self.trick_play_reward

        # Quadratic penalty for distance > 5
        if distance > 5:
            reward -= self.distance_penalty_scale * (distance - 5) ** 2

        # Stack health reward: bonus for balanced stack usage (low gap variance)
        if self.stack_health_scale > 0:
            gaps = [
                (self.stacks[0].top - 2) / 97.0,
                (self.stacks[1].top - 2) / 97.0,
                (99 - self.stacks[2].top) / 98.0,
                (99 - self.stacks[3].top) / 98.0,
            ]
            gap_variance = np.var(gaps)
            reward += self.stack_health_scale * (1.0 - gap_variance)

        # Game phase multiplier: rewards increase late game
        if self.phase_multiplier_scale > 0:
            game_phase = self.total_cards_played / 98.0
            phase_multiplier = 1.0 + self.phase_multiplier_scale * game_phase
            reward *= phase_multiplier

        # Check for victory
        total_cards = len(self.remaining_deck) + sum(len(h) for h in self.hands)
        if total_cards == 0:
            info = {"victory": True, **self._get_episode_stats()}
            return (
                self._get_observation(),
                reward + self.win_reward,
                True,
                False,
                info,
            )

        # Check if turn must end (no more cards or no valid plays)
        can_continue = np.any(self.action_masks())
        must_end_turn = (
            not can_continue or len(self.hands[self.current_player_idx]) == 0
        )

        if must_end_turn:
            if self.cards_played_this_turn < self._min_cards_required():
                info = {"victory": False, "reason": "cannot_play_minimum"}
                info.update(self._get_episode_stats())
                progress_bonus = self.progress_reward_scale * (
                    self.total_cards_played / 98
                )
                return (
                    self._get_observation(),
                    reward - self.loss_penalty + progress_bonus,
                    True,
                    False,
                    info,
                )
            self._end_turn()

            # After ending turn, check if next player can play
            if not np.any(self.action_masks()):
                info = {"victory": False, "reason": "next_player_stuck"}
                info.update(self._get_episode_stats())
                progress_bonus = self.progress_reward_scale * (
                    self.total_cards_played / 98
                )
                return (
                    self._get_observation(),
                    reward - self.loss_penalty + progress_bonus,
                    True,
                    False,
                    info,
                )

        return (
            self._get_observation(),
            reward,
            False,
            False,
            {"action_mask": self.action_masks()},
        )

    def render(self):
        """Print current game state."""
        print(f"\n=== Turn: Player {self.current_player_idx + 1}/{self.n_players} ===")
        print(f"Cards played this turn: {self.cards_played_this_turn}")
        print(f"Deck remaining: {len(self.remaining_deck)}")
        print(
            f"Stack tops: Dec1={self.stacks[0].top}, Dec2={self.stacks[1].top}, "
            f"Inc1={self.stacks[2].top}, Inc2={self.stacks[3].top}"
        )
        print(f"Current hand: {self.hands[self.current_player_idx]}")
        print(f"Valid actions: {np.sum(self.action_masks())}")
