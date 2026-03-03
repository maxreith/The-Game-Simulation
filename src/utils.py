import os
import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(".").resolve() / ".env")

PROJECT_ROOT = Path(__file__).parent
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")


class CardPlay(BaseModel):
    """Represents a single card play action."""

    card: int
    stack: int


class PlayOrder(BaseModel):
    """Represents the ordered list of card plays for a turn."""

    list: list[CardPlay]


_gemini_client = None


def _get_gemini_client():
    """Lazily initialize and return the Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client()
    return _gemini_client


def _load_prompt_template():
    """Load the prompt template from external file."""
    prompt_path = PROJECT_ROOT / "gemini_prompt.txt"
    return prompt_path.read_text()


def _load_rules():
    """Load the game rules from external file."""
    rules_path = PROJECT_ROOT / "game_rules.txt"
    return rules_path.read_text()


class GameOverError(Exception):
    """Raised when a player cannot make a valid move."""

    pass


# =============================================================================
# Stack Index Constants
# =============================================================================
DECREASING_1 = 0
DECREASING_2 = 1
INCREASING_1 = 2
INCREASING_2 = 3


# =============================================================================
# Stack Class with Pre-allocated Arrays
# =============================================================================
class Stack:
    """Pre-allocated stack with O(1) append and top access."""

    MAX_SIZE = 100  # Max cards per stack (98 game cards + initial)

    def __init__(self, initial_value):
        """Initialize with starting card (1 or 99)."""
        self._data = np.empty(self.MAX_SIZE, dtype=np.int32)
        self._data[0] = initial_value
        self._length = 1

    @property
    def top(self):
        """Get the top card. O(1)."""
        return self._data[self._length - 1]

    def push(self, card):
        """Add card to top. O(1). Mutates in-place."""
        self._data[self._length] = card
        self._length += 1

    def __len__(self):
        """Current number of cards."""
        return self._length

    def to_array(self):
        """Return copy of actual contents (for results/debugging)."""
        return self._data[: self._length].copy()

    def copy(self):
        """Create independent copy of this stack."""
        new_stack = Stack.__new__(Stack)
        new_stack._data = self._data.copy()
        new_stack._length = self._length
        return new_stack

    @classmethod
    def from_array(cls, arr):
        """Create a Stack from an existing array (for testing)."""
        stack = cls.__new__(cls)
        stack._data = np.empty(cls.MAX_SIZE, dtype=np.int32)
        stack._data[: len(arr)] = arr
        stack._length = len(arr)
        return stack


def create_stacks(dec1_top=99, dec2_top=99, inc1_top=1, inc2_top=1):
    """Create the 4 game stacks with given initial values."""
    return [
        Stack(dec1_top),  # DECREASING_1 (index 0)
        Stack(dec2_top),  # DECREASING_2 (index 1)
        Stack(inc1_top),  # INCREASING_1 (index 2)
        Stack(inc2_top),  # INCREASING_2 (index 3)
    ]


# =============================================================================
# Helper Functions
# =============================================================================
def play_to_stack(hand, card, chosen_stack, all_stacks):
    """Play a card to a stack by index if valid. Pure function (copies stacks).

    Args:
        hand: Array of cards in the player's hand.
        card: Card value to play.
        chosen_stack: Stack index (0=decreasing_1, 1=decreasing_2, 2=increasing_1, 3=increasing_2).
        all_stacks: List of Stack objects.

    Returns:
        Tuple of (new_hand, new_stacks) after playing the card.
    """
    # Skip if no card is passed
    if isinstance(card, np.ndarray) and len(card) == 0:
        return hand, all_stacks

    stack = all_stacks[chosen_stack]
    top_card = stack.top

    # Check if player has card
    if card not in hand:
        raise ValueError(f"Player does not have card {card}.")

    # Check if card can be played (indices 2,3 are increasing stacks)
    if chosen_stack >= 2:
        can_play = card > top_card or card + 10 == top_card
    else:
        can_play = card < top_card or card - 10 == top_card

    if not can_play:
        raise ValueError(
            f"Card {card} cannot be played on stack {chosen_stack}: Tried to play {card} on top of {top_card}. Hand: {hand}. Stack tops: {[s.top for s in all_stacks]}"
        )

    # Create copies to maintain pure function semantics
    new_stacks = [s.copy() for s in all_stacks]
    new_stacks[chosen_stack].push(card)
    new_hand = hand[hand != card]

    return new_hand, new_stacks


def identify_min_distance_card(hand, stacks):
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


def _build_stack_description(stack_idx, top_value):
    """Build a detailed description of a stack's state and valid moves."""
    if stack_idx < 2:
        direction = "DECREASING"
        valid_range = f"any card < {top_value}"
        reset_value = top_value + 10
        reset_note = f"or exactly {reset_value} (reset)" if reset_value <= 99 else ""
    else:
        direction = "INCREASING"
        valid_range = f"any card > {top_value}"
        reset_value = top_value - 10
        reset_note = f"or exactly {reset_value} (reset)" if reset_value >= 2 else ""

    return f"Stack {stack_idx} ({direction}): top={top_value} → Valid plays: {valid_range} {reset_note}".strip()


def call_api_to_get_play_order(hand, stacks, n_cards_to_play, thinking_level="minimal"):
    """Get play order from Gemini API.

    Args:
        hand: Array of cards in the player's hand.
        stacks: List of Stack objects representing game stacks.
        n_cards_to_play: Minimum number of cards to play this turn.
        thinking_level: Gemini thinking level ("minimal", "low", "medium", "high").

    Returns:
        PlayOrder object with list of card plays.
    """
    stack_descriptions = "\n".join(
        [_build_stack_description(i, stack.top) for i, stack in enumerate(stacks)]
    )

    prompt_template = _load_prompt_template()
    rules = _load_rules()

    prompt = prompt_template.format(
        rules=rules,
        player_hand=hand,
        stack_descriptions=stack_descriptions,
        n_cards_to_play=n_cards_to_play,
    )

    client = _get_gemini_client()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            response_mime_type="application/json",
            response_json_schema=PlayOrder.model_json_schema(),
        ),
    )
    play_order = PlayOrder.model_validate_json(response.text)
    return play_order
