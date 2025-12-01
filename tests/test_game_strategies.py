import numpy as np
import pytest

from game_strategies import _play_to_stack, _reset_pile, _play_lowest_diff, simple_game_strategy, GameOverError


def test_play_to_stack_plays_single_card():
    actual_new_player, actual_new_stack = _play_to_stack(
        player=np.array([10, 20, 30]), 
        card=20, 
        chosen_stack="increasing_stack_1", 
        all_stacks={
        "increasing_stack_1": np.array([1]),})
    expected_new_player = np.array([10, 30])
    expected_new_stack = np.array([1, 20])
    assert np.array_equal(actual_new_player, expected_new_player) # does return expected hand?
    assert np.array_equal(actual_new_stack["increasing_stack_1"], expected_new_stack) # does return expected stack?

def test_play_to_stack_card_missing():
    with pytest.raises(ValueError):
        _play_to_stack(
            player=np.array([10, 20, 30]), 
            card=25, 
            chosen_stack="increasing_stack_1", 
            all_stacks={"increasing_stack_1": np.array([1]),}
            )
        
def test_to_play_stack_invalid_move():
    with pytest.raises(ValueError):
        _play_to_stack(
            player=np.array([10, 20, 30]), 
            card=10, 
            chosen_stack="increasing_stack_1", 
            all_stacks={"increasing_stack_1": np.array([11]),}
            )

def test_reset_pile_with_one_reset_card():
    actual_player, actual_stack = _reset_pile(
        player = np.array([30, 31, 32, 33, 34, 35, 35]),
        stacks = {
        "decreasing_stack_1": np.array([99, 21]),
        "decreasing_stack_2": np.array([99]),
        "increasing_stack_1": np.array([1, 20]),
        "increasing_stack_2": np.array([1])
    })
    expected_player = np.array([30, 32, 33, 34, 35, 35])
    expected_stacks = {
        "decreasing_stack_1": np.array([99, 21, 31]),
        "decreasing_stack_2": np.array([99]),
        "increasing_stack_1": np.array([1, 20]),
        "increasing_stack_2": np.array([1])
    }
    assert np.array_equal(actual_player, expected_player)
    assert all(np.array_equal(actual_stack[k], v) for k, v in expected_stacks.items())

def test_reset_pile_with_many_reset_cards():
    actual_player, actual_stack = _reset_pile(
        player = np.array([31, 41, 51, 10]),
        stacks = {
        "decreasing_stack_1": np.array([99, 21]),
        "decreasing_stack_2": np.array([99]),
        "increasing_stack_1": np.array([1, 20]),
        "increasing_stack_2": np.array([1])
    })
    expected_player = np.array([]) # player plays all cards
    expected_stacks = {
        "decreasing_stack_1": np.array([99, 21, 31, 41, 51]),
        "decreasing_stack_2": np.array([99]),
        "increasing_stack_1": np.array([1, 20, 10]),
        "increasing_stack_2": np.array([1])
    }
    assert np.array_equal(actual_player, expected_player)
    assert all(np.array_equal(actual_stack[k], v) for k, v in expected_stacks.items())

def test_play_lowest_diff_with_normal_hand():
    actual_player, actual_stack = _play_lowest_diff(
        player = np.array([2, 3, 40, 45, 50, 55]),
        stacks = {
        "decreasing_stack_1": np.array([99]),
        "decreasing_stack_2": np.array([99]),
        "increasing_stack_1": np.array([1]),
        "increasing_stack_2": np.array([1])},
        cards_to_play = 2)
    expected_player = np.array([40, 45, 50, 55])
    expected_stacks = {
        "decreasing_stack_1": np.array([99]),
        "decreasing_stack_2": np.array([99]),
        "increasing_stack_1": np.array([1, 2, 3]),
        "increasing_stack_2": np.array([1])}
    assert np.array_equal(actual_player, expected_player)
    assert all(np.array_equal(actual_stack[k], v) for k, v in expected_stacks.items())

def test_play_lowest_diff_with_endgame_hand():
    actual_player, actual_stack = _play_lowest_diff(
        player = np.array([2, 3, 40, 45]),
        stacks = {
        "decreasing_stack_1": np.array([99]),
        "decreasing_stack_2": np.array([99]),
        "increasing_stack_1": np.array([1]),
        "increasing_stack_2": np.array([1])},
        cards_to_play = 1)
    expected_player = np.array([3, 40, 45])
    expected_stacks = {
        "decreasing_stack_1": np.array([99]),
        "decreasing_stack_2": np.array([99]),
        "increasing_stack_1": np.array([1, 2]),
        "increasing_stack_2": np.array([1])}
    assert np.array_equal(actual_player, expected_player)
    assert all(np.array_equal(actual_stack[k], v) for k, v in expected_stacks.items())

def test_simple_game_strategy_reset_and_play():
    actual_player, actual_stack = simple_game_strategy(
        player = np.array([2, 3, 4, 5, 6, 7]),
        stacks = {
        "decreasing_stack_1": np.array([99]),
        "decreasing_stack_2": np.array([99]),
        "increasing_stack_1": np.array([1, 12]),
        "increasing_stack_2": np.array([1])},
        remaining_deck = np.arange(2,99)
    )
    expected_player = np.array([4, 5, 6, 7])
    expected_stacks = {
        "decreasing_stack_1": np.array([99]),
        "decreasing_stack_2": np.array([99]),
        "increasing_stack_1": np.array([1, 12, 2, 3]),
        "increasing_stack_2": np.array([1])}
    assert np.array_equal(expected_player, actual_player)
    assert all(np.array_equal(actual_stack[k], v) for k, v in expected_stacks.items())

def test_simple_game_strategy_game_over():
    with pytest.raises(GameOverError):
        simple_game_strategy(
            player = np.array([4, 5, 6, 7]),
            stacks = {
            "decreasing_stack_1": np.array([99, 2]),
            "decreasing_stack_2": np.array([99, 3]),
            "increasing_stack_1": np.array([1, 97]),
            "increasing_stack_2": np.array([1, 98])},
            remaining_deck = np.arange(2,99)
        )

