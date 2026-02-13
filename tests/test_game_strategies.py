import numpy as np
import pytest

from utils import _play_to_stack, GameOverError, create_stacks
from strategies import bonus_play_strategy, _call_api_to_get_play_order, gemini_strategy


def assert_stack_equals(stacks, expected_array):
    assert any(np.array_equal(expected_array, s.to_array()) for s in stacks)


@pytest.fixture
def empty_stacks():
    return create_stacks(99, 99, 1, 1)


@pytest.fixture
def midgame_stacks():
    return create_stacks(50, 60, 40, 20)


@pytest.fixture
def game_over_stacks():
    return create_stacks(2, 3, 98, 99)


@pytest.fixture
def normal_hand():
    return np.array([2, 22, 40, 45, 51, 57])


def test_play_to_stack_plays_single_card(empty_stacks):
    stacks = empty_stacks
    actual_new_player, actual_new_stacks = _play_to_stack(
        player=np.array([10, 20, 30]), card=20, chosen_stack=2, all_stacks=stacks
    )
    expected_new_player = np.array([10, 30])
    expected_new_stack = np.array([1, 20])
    assert np.array_equal(actual_new_player, expected_new_player)
    assert np.array_equal(actual_new_stacks[2].to_array(), expected_new_stack)


def test_play_to_stack_card_missing(empty_stacks):
    stacks = empty_stacks
    with pytest.raises(ValueError):
        _play_to_stack(
            player=np.array([10, 20, 30]), card=25, chosen_stack=2, all_stacks=stacks
        )


def test_play_to_stack_invalid_move(midgame_stacks, normal_hand):
    stacks, hand = midgame_stacks, normal_hand
    with pytest.raises(ValueError):
        _play_to_stack(
            player=hand,
            card=10,  # card is not in hand
            chosen_stack=2,
            all_stacks=stacks,
        )


@pytest.mark.parametrize(
    "hand,expected_hand,expected_stack",
    [
        # without bonus
        (
            np.array([2, 22, 40, 45, 51, 57]),
            np.array([40, 45, 51, 57]),
            np.array([1, 2, 22]),
        ),
        # with bonus
        (
            np.array([2, 3, 7, 40, 50, 55]),
            np.array([40, 50, 55]),
            np.array([1, 2, 3, 7]),
        ),
    ],
)
def test_bonus_play_strategy(empty_stacks, hand, expected_hand, expected_stack):
    actual_hand, actual_stacks = bonus_play_strategy(hand, empty_stacks)
    assert np.array_equal(actual_hand, expected_hand)
    assert_stack_equals(actual_stacks, expected_stack)


def test_bonus_play_strategy_play_entire_hand(midgame_stacks):
    actual_player, actual_stacks = bonus_play_strategy(
        np.array([38, 27, 17, 7, 8, 12]), midgame_stacks
    )
    expected_player = np.array([38])
    expected_stack = np.array([20, 27, 17, 7, 8, 12])
    assert np.array_equal(actual_player, expected_player)
    assert_stack_equals(actual_stacks, expected_stack)


def test_bonus_play_strategy_game_over(game_over_stacks):
    with pytest.raises(GameOverError):
        bonus_play_strategy(np.array([4, 5, 6, 7]), game_over_stacks)


def test_call_api_to_get_play_order_structure(empty_stacks):
    stacks = empty_stacks
    play_order = _call_api_to_get_play_order(
        player=np.array([10, 20, 30]), stacks=stacks, n_cards_to_play=2
    )
    assert hasattr(play_order, "list")
    assert len(play_order.list) >= 2
    for card_play in play_order.list:
        assert hasattr(card_play, "card")
        assert hasattr(card_play, "stack")


def test_gemini_strategy_plays_reasonable(midgame_stacks):
    actual_player, actual_stacks = gemini_strategy(
        np.array([21, 22]), stacks=midgame_stacks
    )
    expected_player = np.array([])
    expected_stack = np.array([20, 21, 22])
    assert np.array_equal(actual_player, expected_player)
    assert_stack_equals(actual_stacks, expected_stack)


def test_gemini_strategy_game_over(game_over_stacks):
    stacks = game_over_stacks
    with pytest.raises(GameOverError):
        gemini_strategy(
            player=np.array([4, 5, 6, 7]),
            stacks=stacks,
            remaining_deck=np.arange(2, 99),
        )


# @pytest.mark.parametrize("almost_game_over_stacks, hand",
#                          [([4,5,98,99],np.array([2,3,15])),
#                           ([3,4,97,98],np.array([13, 7, 22, 29]))]
#                          )
#
# def test_gemini_avoid_game_over(almost_game_over_stacks, hand):
#     stacks = create_stacks(*almost_game_over_stacks)
#     actual_player, _ = gemini_strategy(hand, stacks)
#     assert len(actual_player) == len(hand) - 2
