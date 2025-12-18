import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel

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
    
    MAX_SIZE: int = 100  # Max cards per stack (98 game cards + initial)
    
    def __init__(self, initial_value: int):
        """Initialize with starting card (1 or 99)."""
        self._data = np.empty(self.MAX_SIZE, dtype=np.int32)
        self._data[0] = initial_value
        self._length = 1
    
    @property
    def top(self) -> int:
        """Get the top card. O(1)."""
        return self._data[self._length - 1]
    
    def push(self, card: int) -> None:
        """Add card to top. O(1). Mutates in-place."""
        self._data[self._length] = card
        self._length += 1
    
    def __len__(self) -> int:
        """Current number of cards."""
        return self._length
    
    def to_array(self) -> np.ndarray:
        """Return copy of actual contents (for results/debugging)."""
        return self._data[:self._length].copy()
    
    def copy(self) -> 'Stack':
        """Create independent copy of this stack."""
        new_stack = Stack.__new__(Stack)
        new_stack._data = self._data.copy()
        new_stack._length = self._length
        return new_stack
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Stack':
        """Create a Stack from an existing array (for testing)."""
        stack = cls.__new__(cls)
        stack._data = np.empty(cls.MAX_SIZE, dtype=np.int32)
        stack._data[:len(arr)] = arr
        stack._length = len(arr)
        return stack


def create_stacks(dec1_top: int = 99, dec2_top: int = 99, 
                  inc1_top: int = 1, inc2_top: int = 1) -> list['Stack']:
    """Create the 4 game stacks with given initial values."""
    return [
        Stack(dec1_top),   # DECREASING_1 (index 0)
        Stack(dec2_top),   # DECREASING_2 (index 1)
        Stack(inc1_top),   # INCREASING_1 (index 2)
        Stack(inc2_top),   # INCREASING_2 (index 3)
    ]


# =============================================================================
# Helper Functions
# =============================================================================
def _play_to_stack(player: np.ndarray, card: int, chosen_stack: int, all_stacks: list[Stack]) -> tuple[np.ndarray, list[Stack]]:
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
        raise ValueError(f"Card {card} cannot be played on stack {chosen_stack}: {card} on top {top_card}. The player's hand: {player}. The other stacks tops: {[s.top for s in all_stacks]}")

    # Create copies to maintain pure function semantics
    new_stacks = [s.copy() for s in all_stacks]
    new_stacks[chosen_stack].push(card)
    new_player = player[player != card]

    return new_player, new_stacks

def _reset_pile(player: np.ndarray, stacks: list[Stack]) -> tuple[np.ndarray, list[Stack]]:
    """
    Play all possible reset cards (±10 jumps) until none remain.
    Keeps checking after each play since new resets may become available.
    
    Stack indices: 0=decreasing_1, 1=decreasing_2, 2=increasing_1, 3=increasing_2
    """
    # Define which stacks use +10 vs -10 for resets: (stack_index, offset)
    stack_offsets = [
        (INCREASING_1, -10),  # Need card = top - 10
        (INCREASING_2, -10),
        (DECREASING_1, +10),  # Need card = top + 10
        (DECREASING_2, +10)
    ]
    
    found_reset = True
    while found_reset:
        found_reset = False
        
        for stack_idx, offset in stack_offsets:
            top_card = stacks[stack_idx].top
            reset_card = top_card + offset
            
            # Check if player has the reset card
            if reset_card in player:
                player, stacks = _play_to_stack(
                    player=player,
                    card=reset_card,
                    chosen_stack=stack_idx,
                    all_stacks=stacks
                )
                found_reset = True
                break  # Restart the loop to check all stacks again
    
    return player, stacks

def _play_lowest_diff(player: np.ndarray, stacks: list[Stack], cards_to_play: int = 2) -> tuple[np.ndarray, list[Stack]]:
    """
    Play cards to stacks where the difference between player's card 
    and stack top is smallest.
    """    
    for _ in range(cards_to_play):
        if len(player) == 0:
            break
        
        tops = np.array([s.top for s in stacks], dtype=np.int32)  # shape (4,)
        player_col = player.reshape(-1, 1)  # shape (n, 1)
        
        diffs = tops - player_col  # shape (n, 4), gives: top - card for all
        
        # Flip sign for increasing stacks (columns 2, 3)
        diffs[:, 2:] = -diffs[:, 2:]  # Now: card - top for increasing
        
        # Mask invalid plays
        diffs[(diffs <= 0) & (diffs != -10)] = 1000
        
        if diffs.min() == 1000:
            break
        
        flat_idx = diffs.argmin()
        best_card_idx, best_stack = divmod(flat_idx, 4)
        best_card = player[best_card_idx]
        
        player, stacks = _play_to_stack(player, best_card, best_stack, stacks)
    
    return player, stacks

# =============================================================================
# Strategy Functions
# =============================================================================

def simple_game_strategy(player, stacks, remaining_deck):
    """First, reset stacks if you can. Then, play cards with minimum distance until you do not have to play a card anymore."""
    n_cards_to_play = 2 if remaining_deck.size > 0 else 1
    
    new_player, new_stacks = _reset_pile(player, stacks)
    new_player, new_stacks = _play_lowest_diff(new_player, new_stacks, n_cards_to_play - (len(player) - len(new_player)))
    
    if len(player) - len(new_player) < n_cards_to_play:
        raise GameOverError(f"Player stuck with {len(new_player)} cards")
    
    return new_player, new_stacks

def bonus_play_strategy(player, stacks, remaining_deck, bonus_play_threshold = 4) -> tuple[np.ndarray, list[Stack]]:
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


def gemini_strategy(player, stacks, remaining_deck):
    """Implementing a strategy that uses Gemini API to determine play order."""
    n_cards_to_play = 2 if len(remaining_deck) > 0 else 1
    play_order = _call_api_to_get_play_order(player, stacks, n_cards_to_play)

    for play in play_order.list:
        player, stacks = _play_to_stack(player, play.card, play.stack, stacks)
                                       
    return player, stacks

def _call_api_to_get_play_order(player: np.ndarray, stacks: list[Stack], n_cards_to_play: int):
    "Get play order from Gemini API."

    prompt = f"""
    You are playing the card game 'The Game'. The rules are as follows:\n{rules}\n\n. Your hand is {player}. 
    The current stacks are {stacks}.\n\n You must play at least {n_cards_to_play} cards. Which cards should you play and on which stacks?            
    """

    class Card_Play(BaseModel):
        card: int
        stack: int

    class Play_Order(BaseModel):
        list : list[Card_Play]

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=f"""
            You are playing the card game 'The Game'. The rules are as follows:\n{rules}\n\n. Your hand is {player}. 
            The current stacks are {stacks}.\n\nBased on this information, which two cards should you play and on which stacks?            
            """,
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            response_mime_type="application/json",
            response_json_schema=Play_Order.model_json_schema(),
            )
    )
    play_order = Play_Order.model_validate_json(response.text)
    return play_order

rules = """
In **"The Game"**, you and your fellow players are not playing against each other; you are playing against the game itself. The goal is to discard all 98 cards in the deck into four specific piles.

Here are the rules:

### 1. The Setup
*   **The Deck:** There are 98 cards numbered **2 through 99**.
*   **The Piles:** There are four piles:
    *   **Two "Ascending" Piles:** These start at **1** and must go **up** (1 $\rightarrow$ 99).
    *   **Two "Descending" Piles:** These start at **100** and must go **down** (100 $\rightarrow$ 2).
    *   **The Decreasing Piles are denoted with integers 0 and 1, and the Ascending Piles with integers 2 and 3.
*   **The Hand:** Depending on the number of players, each person is dealt a hand (e.g., 6 cards for 3–5 players, 7 cards for 2 players, or 8 cards for solo play).

### 2. Gameplay
On your turn, you must follow these two steps:

**Step A: Play Cards (Mandatory)**
You **must** play at least **2 cards** from your hand onto any of the four piles. (If the draw deck is empty, you only need to play 1 card). 
*   On an **Ascending pile (1 $\rightarrow$)**, the card you play must be *higher* than the card currently on top.
*   On a **Descending pile (100 $\rightarrow$)**, the card you play must be *lower* than the card currently on top.
*   You can play as many cards as you want per turn, as long as you play at least two.

**Step B: Draw Cards**
Once you finish playing, draw back up to your original hand size.

### 3. The "Backwards" Rule (The Secret to Winning)
Normally, you must follow the direction of the pile. However, there is one exception that allows you to "reset" a pile:
*   You may play a card in the **wrong direction** if it is **exactly 10 higher or lower** than the top card.
*   *Example:* If the Ascending pile is at **45**, you can play the **35** on top of it to "push" the pile back down, giving your team more room to play.
*   *Example:* If the Descending pile is at **72**, you can play the **82** on top of it.

### 4. Communication
This is a cooperative game, but there is a catch: **You cannot tell other players the specific numbers in your hand.**
*   **Allowed:** "Don't play on this pile," or "I have a really good card for the descending pile."
*   **Not Allowed:** "I have the 44," or "I can play a card that is 2 higher than that 50."

### 5. Winning and Losing
*   **Losing:** If it is a player's turn and they cannot play the minimum required cards (2 cards if the deck exists, 1 if it doesn't), the game ends immediately. Everyone loses!
*   **Winning:** If the team manages to play all 98 cards onto the piles, you have beaten The Game. 
"""