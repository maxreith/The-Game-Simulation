import numpy as np

from game_strategies import GameOverError, Stack, bonus_play_strategy


def _shuffle_cards_custom(card_deck: np.ndarray = None, n_shuffles: int = 200) -> np.ndarray:
    "Shuffles card deck"
    if card_deck is None:
        card_deck = np.arange(2, 100)
    
    shuffled_deck = card_deck.copy()  # Also copy to avoid mutating the input
    for _ in range(n_shuffles):
        random_numbers = np.random.randint(0, len(shuffled_deck), size=2)
        start, end = np.sort(random_numbers)
        shuffled_deck = np.concatenate((
            shuffled_deck[start:end], shuffled_deck[:start], shuffled_deck[end:]
        ))
    return shuffled_deck


def _shuffle_cards(card_deck: np.ndarray = None, n_shuffles: int = 200) -> np.ndarray:
    "Shuffles card deck"
    if card_deck is None:
        card_deck = np.arange(2, 100)
    
    shuffled_deck = card_deck.copy()
    np.random.shuffle(shuffled_deck)
    return shuffled_deck


def _initiate_game(n_players: int, card_deck: np.ndarray, hand_size: int = 6) -> tuple[list[np.ndarray], np.ndarray, list[Stack]]:
    """Create and deal cards to new players."""
    players = []
    card_index = 0
    
    for _ in range(n_players):
        player_cards = card_deck[card_index:card_index + hand_size]
        players.append(player_cards)
        card_index += hand_size

    # Stack order: [decreasing_1, decreasing_2, increasing_1, increasing_2]
    stacks = [
        Stack(99),  # decreasing_1 (index 0)
        Stack(99),  # decreasing_2 (index 1)
        Stack(1),   # increasing_1 (index 2)
        Stack(1)    # increasing_2 (index 3)
    ]
    
    return players, card_deck[card_index:], stacks


def _draw_cards(player: np.ndarray, remaining_deck: np.ndarray, hand_size: int = 6):
    if len(remaining_deck) == 0:
        return player, remaining_deck
    
    cards_to_draw = hand_size - len(player)
    new_player = np.append(player, remaining_deck[:cards_to_draw])
    return new_player, remaining_deck[cards_to_draw:]


def run_game(strategy, n_players: int = 3, n_shuffles: int = 200, bonus_play_threshold: int = 4, use_custom_shuffle: bool = False) -> dict:
    "Runs an instance of the game with a given strategy."
    hand_size = 6 if n_players > 2 else 7
    shuffled_deck = _shuffle_cards_custom(n_shuffles=n_shuffles) if use_custom_shuffle else _shuffle_cards(n_shuffles=n_shuffles)   
    players, remaining_deck, stacks = _initiate_game(n_players, shuffled_deck, hand_size)

    total_cards = lambda: len(remaining_deck) + sum(len(p) for p in players)
    turn = 0
    
    try:
        while total_cards() > 0:
            turn += 1
            if turn > 100:
                raise RuntimeError("Too many turns!")
            
            for i, player in enumerate(players):
                if len(player) == 0:
                    continue
                player, stacks = strategy(player, stacks, remaining_deck, bonus_play_threshold)
                player, remaining_deck = _draw_cards(player, remaining_deck, hand_size)
                players[i] = player
        
        return {"victory": True, "stacks": [s.to_array() for s in stacks], "cards_remaining": 0}
        
    except GameOverError:
        return {"victory": False, "stacks": [s.to_array() for s in stacks], "cards_remaining": total_cards()}


def run_simulation(strategy, n_games: int = 100, n_players: int = 3, bonus_play_threshold: int = 4, n_shuffles: int = 200, use_custom_shuffle: bool = False) -> dict:
    """Run multiple games and collect data."""
    victories = []
    losses = []
    for _ in range(n_games):
        if strategy == bonus_play_strategy:
            result = run_game(strategy, n_players=n_players, bonus_play_threshold=bonus_play_threshold, n_shuffles=n_shuffles, use_custom_shuffle=use_custom_shuffle)
        else:
            result = run_game(strategy, n_players=n_players, n_shuffles=n_shuffles, use_custom_shuffle=use_custom_shuffle)
        
        if result["victory"]:
            victories.append(result)
        else:
            losses.append(result)

    return {
        "victories": victories,
        "losses": losses,
        "win_rate": len(victories) / n_games
    }