import numpy as np

from game_setup import _shuffle_cards, _initiate_game, _draw_cards
from utils import Stack


def test_shuffle_cards_some_shuffling():
    actual = _shuffle_cards(n_shuffles=0)
    assert isinstance(actual, np.ndarray)
    assert len(actual) == 98


def test_initiate_game_player_hands():
    actual_players, _, _ = _initiate_game(n_players=3, card_deck=_shuffle_cards())
    assert isinstance(actual_players, list)
    assert len(actual_players) == 3


def test_initiate_game_remaining_deck():
    n_players = 3
    hand_size = 6
    _, actual_card_deck, _ = _initiate_game(
        n_players=n_players, card_deck=_shuffle_cards()
    )
    assert isinstance(actual_card_deck, np.ndarray)
    assert len(actual_card_deck) == 98 - n_players * hand_size


def test_initiate_game_stacks():
    _, _, actual_stacks = _initiate_game(n_players=3, card_deck=_shuffle_cards())
    assert isinstance(actual_stacks, list)
    assert len(actual_stacks) == 4
    assert all(isinstance(s, Stack) for s in actual_stacks)
    assert actual_stacks[0].top == 99  # decreasing_1
    assert actual_stacks[1].top == 99  # decreasing_2
    assert actual_stacks[2].top == 1  # increasing_1
    assert actual_stacks[3].top == 1  # increasing_2


def test_draw_cards_normal_turn():
    actual_hand, actual_deck = _draw_cards(
        hand=np.array([10, 20, 30]),
        remaining_deck=np.array([40, 50, 60, 70, 80]),
        hand_size=6,
    )
    expected_hand = np.array([10, 20, 30, 40, 50, 60])
    expected_deck = np.array([70, 80])
    assert np.array_equal(actual_hand, expected_hand)
    assert np.array_equal(actual_deck, expected_deck)


def test_draw_cards_empty_deck():
    actual_hand, actual_deck = _draw_cards(
        hand=np.array([10, 20, 30]), remaining_deck=np.array([]), hand_size=6
    )
    expected_hand = np.array([10, 20, 30])
    expected_deck = np.array([])
    assert np.array_equal(actual_hand, expected_hand)
    assert np.array_equal(actual_deck, expected_deck)
