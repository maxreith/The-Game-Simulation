import numpy as np

from utils import (
    GameOverError,
    play_to_stack,
    call_api_to_get_play_order,
    identify_min_distance_card,
)


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
        best_card, best_stack, min_diff = identify_min_distance_card(hand, stacks)
        if i >= n_cards_to_play and min_diff > bonus_play_threshold:
            break
        hand, stacks = play_to_stack(hand, best_card, best_stack, stacks)

    return hand, stacks


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
    play_order = call_api_to_get_play_order(
        hand, stacks, n_cards_to_play, thinking_level
    )

    for play in play_order.list:
        try:
            hand, stacks = play_to_stack(hand, play.card, play.stack, stacks)
        except ValueError as e:
            raise GameOverError(
                f"""Gemini requested an invalid play. Tried to play card {getattr(play, "card", None)} on stack {getattr(play, "stack", None)}
                with hand {hand} and stack tops {[s.top for s in stacks]}."""
            ) from e

    if len(play_order.list) < n_cards_to_play:
        raise GameOverError(f"Player stuck with {len(hand)} cards")

    return hand, stacks
