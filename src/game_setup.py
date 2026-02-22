import numpy as np

from utils import GameOverError, Stack


def _shuffle_cards_custom(card_deck=None, n_shuffles=200):
    "Shuffles card deck"
    if card_deck is None:
        card_deck = np.arange(2, 100)

    shuffled_deck = card_deck.copy()  # Also copy to avoid mutating the input
    for _ in range(n_shuffles):
        random_numbers = np.random.randint(0, len(shuffled_deck), size=2)
        start, end = np.sort(random_numbers)
        shuffled_deck = np.concatenate(
            (shuffled_deck[start:end], shuffled_deck[:start], shuffled_deck[end:])
        )
    return shuffled_deck


def _shuffle_cards(card_deck=None, n_shuffles=200):
    "Shuffles card deck"
    if card_deck is None:
        card_deck = np.arange(2, 100)

    shuffled_deck = card_deck.copy()
    np.random.shuffle(shuffled_deck)
    return shuffled_deck


def _initiate_game(n_players, card_deck, hand_size=6):
    """Create and deal cards to players.

    Args:
        n_players: Number of players in the game.
        card_deck: Shuffled deck of cards.
        hand_size: Number of cards per player.

    Returns:
        Tuple of (hands, remaining_deck, stacks).
    """
    hands = []
    card_index = 0

    for _ in range(n_players):
        hand_cards = card_deck[card_index : card_index + hand_size]
        hands.append(hand_cards)
        card_index += hand_size

    # Stack order: [decreasing_1, decreasing_2, increasing_1, increasing_2]
    stacks = [
        Stack(99),  # decreasing_1 (index 0)
        Stack(99),  # decreasing_2 (index 1)
        Stack(1),  # increasing_1 (index 2)
        Stack(1),  # increasing_2 (index 3)
    ]

    return hands, card_deck[card_index:], stacks


def _draw_cards(hand, remaining_deck, hand_size=6):
    """Draw cards from deck to refill hand.

    Args:
        hand: Array of cards currently in the player's hand.
        remaining_deck: Cards remaining in the deck.
        hand_size: Target hand size.

    Returns:
        Tuple of (new_hand, remaining_deck).
    """
    if len(remaining_deck) == 0:
        return hand, remaining_deck

    cards_to_draw = hand_size - len(hand)
    new_hand = np.append(hand, remaining_deck[:cards_to_draw])
    return new_hand, remaining_deck[cards_to_draw:]


def run_game(strategy, n_players=3, n_shuffles=200, use_custom_shuffle=False):
    """Run an instance of the game with a given strategy.

    Args:
        strategy: A callable with signature (hand, stacks, remaining_deck) -> (hand, stacks).
            Strategy-specific parameters should be pre-configured via functools.partial.
        n_players: Number of players in the game.
        n_shuffles: Number of shuffles for the deck.
        use_custom_shuffle: Whether to use the custom shuffle algorithm.

    Returns:
        A dict with victory status, final stacks, and cards remaining.
    """
    hand_size = 6 if n_players > 2 else 7
    shuffled_deck = (
        _shuffle_cards_custom(n_shuffles=n_shuffles)
        if use_custom_shuffle
        else _shuffle_cards(n_shuffles=n_shuffles)
    )
    hands, remaining_deck, stacks = _initiate_game(n_players, shuffled_deck, hand_size)

    turn = 0

    try:
        while len(remaining_deck) + sum(len(h) for h in hands) > 0:
            turn += 1
            if turn > 100:
                raise RuntimeError("Too many turns!")

            for i, hand in enumerate(hands):
                if len(hand) == 0:
                    continue
                hand, stacks = strategy(hand, stacks, remaining_deck)
                hand, remaining_deck = _draw_cards(hand, remaining_deck, hand_size)
                hands[i] = hand

        return {
            "victory": True,
            "stacks": [s.to_array() for s in stacks],
            "cards_remaining": 0,
            "turns": turn,
        }

    except GameOverError:
        cards_remaining = len(remaining_deck) + sum(len(h) for h in hands)
        return {
            "victory": False,
            "stacks": [s.to_array() for s in stacks],
            "cards_remaining": cards_remaining,
            "turns": turn,
        }


def run_simulation(
    strategy, n_games=100, n_players=3, n_shuffles=200, use_custom_shuffle=False
):
    """Run multiple games and collect data.

    Args:
        strategy: A callable with signature (hand, stacks, remaining_deck) -> (hand, stacks).
            Strategy-specific parameters should be pre-configured via functools.partial.
        n_games: Number of games to simulate.
        n_players: Number of players in each game.
        n_shuffles: Number of shuffles for the deck.
        use_custom_shuffle: Whether to use the custom shuffle algorithm.

    Returns:
        A dict with victories list, losses list, and win_rate.
    """
    victories = []
    losses = []
    for _ in range(n_games):
        result = run_game(
            strategy,
            n_players=n_players,
            n_shuffles=n_shuffles,
            use_custom_shuffle=use_custom_shuffle,
        )

        if result["victory"]:
            victories.append(result)
        else:
            losses.append(result)

    return {
        "victories": victories,
        "losses": losses,
        "win_rate": len(victories) / n_games,
    }
