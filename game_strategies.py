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
# Helper Functions
# =============================================================================
def _play_to_stack(player: np.ndarray, card: int, chosen_stack: int, all_stacks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Play a card to a stack by index if valid. Pure function.
    
    Stack indices: 0=decreasing_1, 1=decreasing_2, 2=increasing_1, 3=increasing_2
    """
    # Skip if no card is passed
    if isinstance(card, np.ndarray) and len(card) == 0:
        return player, all_stacks

    stack = all_stacks[chosen_stack]

    # Check if player has card
    if card not in player:
        raise ValueError(f"Player does not have card {card}.")

    # Check if card can be played (indices 2,3 are increasing stacks)
    if chosen_stack >= 2:
        can_play = card > stack[-1] or card + 10 == stack[-1]
    else:
        can_play = card < stack[-1] or card - 10 == stack[-1]

    if not can_play:
        raise ValueError(f"Card {card} cannot be played on stack {chosen_stack}.")

    # Create new objects instead of mutating
    new_stacks = [
        np.append(s, card) if i == chosen_stack else s
        for i, s in enumerate(all_stacks)
    ]
    new_player = player[player != card]

    return new_player, new_stacks

def _reset_pile(player: np.ndarray, stacks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
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
            top_card = stacks[stack_idx][-1]
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

def _play_lowest_diff(player: np.ndarray, stacks: list[np.ndarray], cards_to_play: int = 2, hand_size: int = 6) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Play cards to stacks where the difference between player's card 
    and stack top is smallest. Stops when player has (hand_size - cards_to_play) cards.
    Pure function.
    
    Stack indices: 0=decreasing_1, 1=decreasing_2, 2=increasing_1, 3=increasing_2
    """    
    for _ in range(cards_to_play):
        if len(player) == 0:
            break
        
        # Find all valid (card, stack_idx, diff) combinations
        valid_plays = [
            (card, stack_idx, card - stack[-1] if stack_idx >= 2 else stack[-1] - card)
            for card in player
            for stack_idx, stack in enumerate(stacks)
            if (stack_idx >= 2 and card > stack[-1]) or
               (stack_idx < 2 and card < stack[-1])
        ]
        
        if not valid_plays:
            break
        
        best_card, best_stack, _ = min(valid_plays, key=lambda x: x[2])
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


