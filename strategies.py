import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv

from utils import GameOverError, Stack, _play_to_stack

load_dotenv(Path(".").resolve() / ".env")


def bonus_play_strategy(player: np.ndarray, stacks: list[Stack], remaining_deck: np.ndarray = np.arange(2, 99), bonus_play_threshold = 4) -> tuple[np.ndarray, list[Stack]]:
    """
    """
    n_cards_to_play = 2 if len(remaining_deck) > 0 else 1

    original_hand_size = len(player)

    tops = np.array([s.top for s in stacks], dtype=np.int32)
    player_col = player.reshape(-1, 1)
    diffs = tops - player_col

    diffs[:, 2:] = -diffs[:, 2:]

    for i in range(len(player)):
        # Create playable_diffs from diffs for card selection
        playable_diffs = diffs.copy()
        playable_diffs[(playable_diffs <= 0) & (playable_diffs != -10)] = 1000

        min_diff = playable_diffs.min()

        # Stop after n_cards_to_play if diff exceeds threshold
        if i >= n_cards_to_play and min_diff > bonus_play_threshold or min_diff == 1000:
            break

        flat_idx = playable_diffs.argmin()
        best_card_idx, best_stack = divmod(flat_idx, 4)
        best_card = player[best_card_idx]

        player, stacks = _play_to_stack(player, best_card, best_stack, stacks)

        diffs = np.delete(diffs, best_card_idx, axis=0)

        new_top = stacks[best_stack].top
        if best_stack < 2:  # Decreasing stack
            diffs[:, best_stack] = new_top - player
        else:  # Increasing stack (needs sign flip)
            diffs[:, best_stack] = -(new_top - player)
        tops[best_stack] = new_top

    n_cards_played = original_hand_size - len(player)
    if n_cards_played < n_cards_to_play:
        raise GameOverError(f"Player stuck with {len(player)} cards")

    return player, stacks


def gemini_strategy(player: np.ndarray, stacks: list[Stack], remaining_deck: np.ndarray = np.arange(2, 99)) -> tuple[np.ndarray, list[Stack]]:
    """Implementing a strategy that uses Gemini API to determine play order."""
    n_cards_to_play = 2 if len(remaining_deck) > 0 else 1
    play_order = _call_api_to_get_play_order(player, stacks, n_cards_to_play)

    for play in play_order.list:
        try:
            player, stacks = _play_to_stack(player, play.card, play.stack, stacks)
        except ValueError as e:
            raise GameOverError(
                f"""Gemini requested an invalid play. Tried to play card {getattr(play, 'card', None)} on stack {getattr(play, 'stack', None)}
                with player hand {player} and stack tops {[s.top for s in stacks]}."""
            ) from e

    if len(play_order.list) < n_cards_to_play:
        raise GameOverError(f"Player stuck with {len(player)} cards")

    return player, stacks


def _call_api_to_get_play_order(player: np.ndarray, stacks: list[Stack], n_cards_to_play: int):
    """Get play order from Gemini API."""
    #import pdbp; breakpoint()

    stack_descriptions = "\n".join([f"Stack {i}: top = {stack.top}" for i, stack in enumerate(stacks)])

    prompt = f"""
    You are playing the card game 'The Game'. Your hand is {player}.
    The current stacks and their top cards are {stack_descriptions}.\n\n
    Note that decreasing piles are identified with integers 0 and 1, and increasing piles with integers 2 and 3. Thus, to play a card on the first decreasing pile, you would specify stack 0. To play on the second increasing pile, you would specify stack 3.
    You must play at least {n_cards_to_play} cards from your hand. You must avoid invalid play, if you can. Play stack resets if possible. Which cards should you play and on which stacks?\n
    """
    class Card_Play(BaseModel):
        card: int
        stack: int

    class Play_Order(BaseModel):
        list : list[Card_Play]

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            response_mime_type="application/json",
            response_json_schema=Play_Order.model_json_schema(),
            )
    )
    play_order = Play_Order.model_validate_json(response.text)
    return play_order
