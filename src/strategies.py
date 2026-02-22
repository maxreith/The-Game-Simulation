import numpy as np

from utils import GameOverError, _play_to_stack, _call_api_to_get_play_order


def bonus_play_strategy(
    hand, stacks, remaining_deck=np.arange(2, 99), bonus_play_threshold=4
):
    """Play cards with minimum distance to stack tops, with optional bonus plays.

    Args:
        hand: Array of cards in the player's hand.
        stacks: List of Stack objects representing game stacks.
        remaining_deck: Cards remaining in the deck.
        bonus_play_threshold: Maximum distance for bonus plays beyond minimum.

    Returns:
        Tuple of (updated hand, updated stacks).
    """
    n_cards_to_play = 2 if len(remaining_deck) > 0 else 1
    original_hand_size = len(hand)

    for i in range(original_hand_size):
        best_card, best_stack, min_diff = _identify_min_distance_card(hand, stacks)
        if i >= n_cards_to_play and min_diff > bonus_play_threshold:
            break
        hand, stacks = _play_to_stack(hand, best_card, best_stack, stacks)

    return hand, stacks


def _identify_min_distance_card(hand, stacks):
    """Identify the best card and stack to play on.

    Args:
        hand: Array of cards in the player's hand.
        stacks: List of Stack objects representing game stacks.

    Returns:
        Tuple of (best_card, best_stack_index, minimum_distance).
    """
    tops = np.array([s.top for s in stacks], dtype=np.int32)
    hand_col = hand.reshape(-1, 1)
    diffs = tops - hand_col
    diffs[:, 2:] = -diffs[:, 2:]

    playable_diffs = diffs.copy()
    playable_diffs[(playable_diffs <= 0) & (playable_diffs != -10)] = 1000
    flat_idx = playable_diffs.argmin()
    best_card_idx, best_stack = divmod(flat_idx, 4)
    best_card = hand[best_card_idx]

    min_diff = playable_diffs[best_card_idx, best_stack]

    if min_diff == 1000:
        raise GameOverError(f"No playable card found in hand {hand}")

    return best_card, best_stack, min_diff


def gemini_strategy(
    hand, stacks, remaining_deck=np.arange(2, 99), thinking_level="minimal"
):
    """Use Gemini API to determine play order.

    Args:
        hand: Array of cards in the player's hand.
        stacks: List of Stack objects representing game stacks.
        remaining_deck: Cards remaining in the deck.
        thinking_level: Gemini thinking level ("minimal", "low", "medium", "high").

    Returns:
        Tuple of (updated hand, updated stacks).
    """
    n_cards_to_play = 2 if len(remaining_deck) > 0 else 1
    play_order = _call_api_to_get_play_order(
        hand, stacks, n_cards_to_play, thinking_level
    )

    for play in play_order.list:
        try:
            hand, stacks = _play_to_stack(hand, play.card, play.stack, stacks)
        except ValueError as e:
            raise GameOverError(
                f"""Gemini requested an invalid play. Tried to play card {getattr(play, "card", None)} on stack {getattr(play, "stack", None)}
                with hand {hand} and stack tops {[s.top for s in stacks]}."""
            ) from e

    if len(play_order.list) < n_cards_to_play:
        raise GameOverError(f"Player stuck with {len(hand)} cards")

    return hand, stacks
