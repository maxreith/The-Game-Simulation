import numpy as np
import pytest

from game_strategies import (
    _play_to_stack, bonus_play_strategy, _call_api_to_get_play_order, gemini_strategy, GameOverError,
    DECREASING_1, DECREASING_2, INCREASING_1, INCREASING_2, Stack, create_stacks
)

@pytest.fixture
def empty_stacks():
    """Four fresh stacks: two decreasing at 99, two increasing at 1."""
    return [Stack(99), Stack(99), Stack(1), Stack(1)]

@pytest.fixture
def midgame_stacks():
    """Stacks in a mid-game state."""
    return [
        Stack.from_array(np.array([99, 70, 50])),  
        Stack.from_array(np.array([99, 60])),         
        Stack.from_array(np.array([1, 25, 40])),    
        Stack.from_array(np.array([1, 20]))    
    ]

@pytest.fixture
def game_over_stacks():
    """Stacks in a game-over state."""
    return [
        Stack.from_array(np.array([99, 2])),  
        Stack.from_array(np.array([99, 3])),         
        Stack.from_array(np.array([1, 97])),    
        Stack.from_array(np.array([1, 98]))    
    ]

@pytest.fixture
def hand_normal():
    """A normal hand used across several tests."""
    return np.array([2, 22, 40, 45, 51, 57])

def test_play_to_stack_plays_single_card(empty_stacks):
    stacks = empty_stacks
    actual_new_player, actual_new_stacks = _play_to_stack(
        player=np.array([10, 20, 30]), 
        card=20, 
        chosen_stack=INCREASING_1, 
        all_stacks=stacks
    )
    expected_new_player = np.array([10, 30])
    expected_new_stack = np.array([1, 20])
    assert np.array_equal(actual_new_player, expected_new_player)
    assert np.array_equal(actual_new_stacks[INCREASING_1].to_array(), expected_new_stack)

def test_play_to_stack_card_missing(empty_stacks):
    stacks = empty_stacks
    with pytest.raises(ValueError):
        _play_to_stack(
            player=np.array([10, 20, 30]), 
            card=25, 
            chosen_stack=INCREASING_1, 
            all_stacks=stacks
        )
        
def test_play_to_stack_invalid_move(midgame_stacks):
    stacks = midgame_stacks
    with pytest.raises(ValueError):
        _play_to_stack(
            player=np.array([10, 20, 30]), 
            card=10, 
            chosen_stack=INCREASING_1, 
            all_stacks=stacks
        )





def test_bonus_play_strategy_play_wo_bonus(empty_stacks):
    stacks = empty_stacks
    actual_player, actual_stacks = bonus_play_strategy(
        player=np.array([2, 3, 40, 45, 50, 55]),
        stacks=stacks,
        remaining_deck=np.arange(2, 99),
        bonus_play_threshold=5
    )
    expected_player = np.array([40, 45, 50, 55])
    expected_stacks = [
        np.array([99]),      # decreasing_1
        np.array([99]),      # decreasing_2
        np.array([1, 2, 3]), # increasing_1
        np.array([1])        # increasing_2
    ]
    assert np.array_equal(actual_player, expected_player)
    assert all(np.array_equal(actual_stacks[i].to_array(), expected_stacks[i]) for i in range(4))

def test_bonus_play_strategy_play_with_bonus(empty_stacks):
    stacks = empty_stacks
    actual_player, actual_stacks = bonus_play_strategy(
        player=np.array([2, 3, 7, 40, 50, 55]),
        stacks=stacks,
        remaining_deck=np.arange(2, 99),
        bonus_play_threshold=5
    )
    expected_player = np.array([40, 50, 55])
    expected_stacks = [
        np.array([99]),          # decreasing_1
        np.array([99]),          # decreasing_2
        np.array([1, 2, 3, 7]), # increasing_1
        np.array([1])            # increasing_2
    ]
    assert np.array_equal(actual_player, expected_player)
    assert all(np.array_equal(actual_stacks[i].to_array(), expected_stacks[i]) for i in range(4))

def test_bonus_play_strategy_play_play_entire_hand(midgame_stacks):
    stacks = midgame_stacks
    actual_player, actual_stacks = bonus_play_strategy(
        player=np.array([38, 27, 17, 7, 8, 12]),
        stacks=stacks,
        remaining_deck=np.arange(2, 99),
        bonus_play_threshold=5
    )
    expected_player = np.array([38])
    expected_stacks = [
        np.array([99, 70, 50]),  # decreasing_1
        np.array([99, 60]),          # decreasing_2
        np.array([1, 25, 40]),    # increasing_1
        np.array([1, 20, 27, 17, 7, 8, 12]) # increasing_2
    ]
    assert np.array_equal(actual_player, expected_player)
    assert all(np.array_equal(actual_stacks[i].to_array(), expected_stacks[i]) for i in range(4))

def test_bonus_play_strategy_game_over(game_over_stacks):
    stacks = game_over_stacks
    with pytest.raises(GameOverError):
        bonus_play_strategy(
            player=np.array([4, 5, 6, 7]),
            stacks=stacks,
            remaining_deck=np.arange(2, 99),
            bonus_play_threshold=5
        )

def test_call_api_to_get_play_order_structure(empty_stacks):
    stacks = empty_stacks
    play_order = _call_api_to_get_play_order(
        player=np.array([10, 20, 30]),
        stacks=stacks,
        n_cards_to_play=2
    )
    assert hasattr(play_order, 'list')
    assert len(play_order.list) >= 2
    for card_play in play_order.list:
        assert hasattr(card_play, 'card')
        assert hasattr(card_play, 'stack')

def test_gemini_strategy_plays_reasonable(midgame_stacks):
    stacks = midgame_stacks
    actual_player, actual_stacks = gemini_strategy(
        player=np.array([21, 22]),
        stacks=stacks,
        remaining_deck=np.arange(2, 99)
    )
    expected_player = np.array([])
    expected_stacks = [
        np.array([99, 70, 50]),  # decreasing_1
        np.array([99, 60]),          # decreasing_2
        np.array([1, 25, 40]),    # increasing_1
        np.array([1, 20, 21, 22]) # increasing_2
    ]
    assert np.array_equal(actual_player, expected_player)
    assert np.array_equal(actual_stacks[0].to_array(), expected_stacks[0])
    assert np.array_equal(actual_stacks[1].to_array(), expected_stacks[1])
    assert np.array_equal(actual_stacks[2].to_array(), expected_stacks[2])
    assert np.array_equal(actual_stacks[3].to_array(), expected_stacks[3])

def test_gemini_strategy_game_over(game_over_stacks):
    stacks = game_over_stacks
    with pytest.raises(GameOverError):
        gemini_strategy(
            player=np.array([4, 5, 6, 7]),
            stacks=stacks,
            remaining_deck=np.arange(2, 99)
        )

@pytest.mark.parametrize("almost_game_over_stacks, hand", 
                         [([4,5,98,99],np.array([2,3,15])),
                          ([3,4,97,98],np.array([13, 7, 22, 29]))]
                         )

def test_gemini_avoid_game_over(almost_game_over_stacks, hand):
    stacks = create_stacks(*almost_game_over_stacks)
    actual_player, _ = gemini_strategy(hand, stacks)
    assert len(actual_player) == len(hand) - 2 # Should play two cards to avoid game over