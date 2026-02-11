import numpy as np

from utils import GameOverError, _play_to_stack, _call_api_to_get_play_order


def bonus_play_strategy(
    player, stacks, remaining_deck=np.arange(2, 99), bonus_play_threshold=4
):
    "Computes the card with minimum distance to all stacks. Plays if there are still cards to play, or if the best card is within the bonus_play_threshold."
    n_cards_to_play = 2 if len(remaining_deck) > 0 else 1
    original_hand_size = len(player)

    for i in range(original_hand_size):
        best_card, best_stack, min_diff = _identify_min_distance_card(player, stacks)
        if i >= n_cards_to_play and min_diff > bonus_play_threshold:
            break
        player, stacks = _play_to_stack(player, best_card, best_stack, stacks)

    return player, stacks


def _identify_min_distance_card(player, stacks):
    """Identify the best card and stack to play on."""
    tops = np.array([s.top for s in stacks], dtype=np.int32)
    player_col = player.reshape(-1, 1)
    diffs = tops - player_col
    diffs[:, 2:] = -diffs[:, 2:]

    playable_diffs = diffs.copy()
    playable_diffs[(playable_diffs <= 0) & (playable_diffs != -10)] = 1000
    flat_idx = playable_diffs.argmin()
    best_card_idx, best_stack = divmod(flat_idx, 4)
    best_card = player[best_card_idx]

    min_diff = playable_diffs[best_card_idx, best_stack]

    if min_diff == 1000:
        raise GameOverError(f"No playable card found in hand {player}")

    return best_card, best_stack, min_diff


def gemini_strategy(player, stacks, remaining_deck=np.arange(2, 99)):
    """Implementing a strategy that uses Gemini API to determine play order."""
    n_cards_to_play = 2 if len(remaining_deck) > 0 else 1
    play_order = _call_api_to_get_play_order(player, stacks, n_cards_to_play)

    for play in play_order.list:
        try:
            player, stacks = _play_to_stack(player, play.card, play.stack, stacks)
        except ValueError as e:
            raise GameOverError(
                f"""Gemini requested an invalid play. Tried to play card {getattr(play, "card", None)} on stack {getattr(play, "stack", None)}
                with player hand {player} and stack tops {[s.top for s in stacks]}."""
            ) from e

    if len(play_order.list) < n_cards_to_play:
        raise GameOverError(f"Player stuck with {len(player)} cards")

    return player, stacks
