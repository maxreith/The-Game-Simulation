import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(".").resolve() / ".env")


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
def _play_to_stack(player, card, chosen_stack, all_stacks):
    """Play a card to a stack by index if valid. Pure function (copies stacks).

    Stack indices: 0=decreasing_1, 1=decreasing_2, 2=increasing_1, 3=increasing_2
    """
    # Skip if no card is passed
    if isinstance(card, np.ndarray) and len(card) == 0:
        return player, all_stacks

    stack = all_stacks[chosen_stack]
    top_card = stack.top

    # Check if player has card
    if card not in player:
        raise ValueError(f"Player does not have card {card}.")

    # Check if card can be played (indices 2,3 are increasing stacks)
    if chosen_stack >= 2:
        can_play = card > top_card or card + 10 == top_card
    else:
        can_play = card < top_card or card - 10 == top_card

    if not can_play:
        raise ValueError(
            f"Card {card} cannot be played on stack {chosen_stack}: Tried to play {card} on top of {top_card}. The player's hand: {player}. The other stacks tops: {[s.top for s in all_stacks]}"
        )

    # Create copies to maintain pure function semantics
    new_stacks = [s.copy() for s in all_stacks]
    new_stacks[chosen_stack].push(card)
    new_player = player[player != card]

    return new_player, new_stacks


def _call_api_to_get_play_order(player, stacks, n_cards_to_play):
    """Get play order from Gemini API."""
    # import pdbp; breakpoint()

    stack_descriptions = "\n".join(
        [f"Stack {i}: top = {stack.top}" for i, stack in enumerate(stacks)]
    )

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
        list: list[Card_Play]

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            response_mime_type="application/json",
            response_json_schema=Play_Order.model_json_schema(),
        ),
    )
    play_order = Play_Order.model_validate_json(response.text)
    return play_order


rules = """
In **"The Game"**, you and your fellow players are not playing against each other; you are playing against the game itself. The goal is to discard all 98 cards in the deck into four specific piles.

Here are the rules:

### 1. The Setup
*   **The Deck:** There are 98 cards numbered **2 through 99**.
*   **The Piles:** There are four piles:
    *   **Two "Ascending" Piles:** These start at **1** and must go **up** (1 $\\rightarrow$ 99).
    *   **Two "Descending" Piles:** These start at **100** and must go **down** (100 $\\rightarrow$ 2).
*   **The Hand:** Depending on the number of players, each person is dealt a hand (e.g., 6 cards for 3–5 players, 7 cards for 2 players, or 8 cards for solo play).

### 2. Gameplay
On your turn, you must follow these two steps:

**Step A: Play Cards (Mandatory)**
You **must** play at least **2 cards** from your hand onto any of the four piles. (If the draw deck is empty, you only need to play 1 card).
*   On an **Ascending pile (1 $\\rightarrow$)**, the card you play must be *higher* than the card currently on top.
*   On a **Descending pile (100 $\\rightarrow$)**, the card you play must be *lower* than the card currently on top.
*   You can play as many cards as you want per turn, as long as you play at least two.

**Step B: Draw Cards**
Once you finish playing, draw back up to your original hand size.

### 3. The "Backwards" Rule (The Secret to Winning)
Normally, you must follow the direction of the pile. However, there is one exception that allows you to "reset" a pile:
*   You may play a card in the **wrong direction** if it is **exactly 10 higher or lower** than the top card.
*   *Example:* If the Ascending pile is at **45**, you can play the **35** on top of it to "push" the pile back down, giving your team more room to play.
*   *Example:* If the Descending pile is at **72**, you can play the **82** on top of it.

### 4. Winning and Losing
*   **Losing:** If it is a player's turn and they cannot play the minimum required cards (2 cards if the deck exists, 1 if it doesn't), the game ends immediately. Everyone loses!
*   **Winning:** If the team manages to play all 98 cards onto the piles, you have beaten The Game.
"""
