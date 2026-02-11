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
    """Create and deal cards to new players."""
    players = []
    card_index = 0

    for _ in range(n_players):
        player_cards = card_deck[card_index : card_index + hand_size]
        players.append(player_cards)
        card_index += hand_size

    # Stack order: [decreasing_1, decreasing_2, increasing_1, increasing_2]
    stacks = [
        Stack(99),  # decreasing_1 (index 0)
        Stack(99),  # decreasing_2 (index 1)
        Stack(1),  # increasing_1 (index 2)
        Stack(1),  # increasing_2 (index 3)
    ]

    return players, card_deck[card_index:], stacks


def _draw_cards(player, remaining_deck, hand_size=6):
    if len(remaining_deck) == 0:
        return player, remaining_deck

    cards_to_draw = hand_size - len(player)
    new_player = np.append(player, remaining_deck[:cards_to_draw])
    return new_player, remaining_deck[cards_to_draw:]


def run_game(strategy, n_players=3, n_shuffles=200, use_custom_shuffle=False):
    """Runs an instance of the game with a given strategy.

    Args:
        strategy: A callable with signature (player, stacks, remaining_deck) -> (player, stacks).
            Strategy-specific parameters should be pre-configured via functools.partial.
        n_players: Number of players in the game.
        n_shuffles: Number of shuffles for the deck.
        use_custom_shuffle: Whether to use the custom shuffle algorithm.

    Returns:
        A dict with victory status, final stacks, and cards remaining.
    """
    hand_size = 6 if n_players > 2 else 7
    shuffled_deck = _shuffle_cards_custom(n_shuffles=n_shuffles) if use_custom_shuffle else _shuffle_cards(n_shuffles=n_shuffles)
    players, remaining_deck, stacks = _initiate_game(n_players, shuffled_deck, hand_size)

    turn = 0

    try:
        while len(remaining_deck) + sum(len(p) for p in players) > 0:
            turn += 1
            if turn > 100:
                raise RuntimeError("Too many turns!")

            for i, player in enumerate(players):
                if len(player) == 0:
                    continue
                player, stacks = strategy(player, stacks, remaining_deck)
                player, remaining_deck = _draw_cards(player, remaining_deck, hand_size)
                players[i] = player

        return {
            "victory": True,
            "stacks": [s.to_array() for s in stacks],
            "cards_remaining": 0,
        }

    except GameOverError:
        cards_remaining = len(remaining_deck) + sum(len(p) for p in players)
        return {
            "victory": False,
            "stacks": [s.to_array() for s in stacks],
            "cards_remaining": cards_remaining,
        }


def run_simulation(strategy, n_games=100, n_players=3, n_shuffles=200, use_custom_shuffle=False):
    """Run multiple games and collect data.

    Args:
        strategy: A callable with signature (player, stacks, remaining_deck) -> (player, stacks).
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
