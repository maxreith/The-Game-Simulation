import numpy as np

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
        raise ValueError(f"Card {card} cannot be played on stack {chosen_stack}.")

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

def _play_lowest_diff(player: np.ndarray, stacks: list[Stack], cards_to_play: int = 2, hand_size: int = 6) -> tuple[np.ndarray, list[Stack]]:
    """
    Play cards to stacks where the difference between player's card 
    and stack top is smallest.
    """    
    for _ in range(cards_to_play):
        if len(player) == 0:
            break
        
        # Cache stack tops as numpy array for vectorized operations
        tops = np.array([s.top for s in stacks], dtype=np.int32)
        
        diffs = np.empty((len(player), 4), dtype=np.int32)
        diffs[:, 0] = tops[0] - player  # decreasing
        diffs[:, 1] = tops[1] - player  # decreasing
        diffs[:, 2] = player - tops[2]  # increasing
        diffs[:, 3] = player - tops[3]  # increasing
        
        # Valid plays have positive difference (card can be placed)
        valid_mask = diffs > 0
        
        if not valid_mask.any():
            break
        
        # Set invalid plays to large number so they're not selected
        diffs = np.where(diffs > 0, diffs, 1000)
        
        # Find minimum difference position
        flat_idx = np.argmin(diffs)
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

def strategy(player, stacks, remaining_deck):
    """
    """
    n_cards_to_play = 2 if len(remaining_deck) > 0 else 1

    new_player, new_stacks = _reset_pile(player, stacks)


