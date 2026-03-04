"""Tests for validating example game files."""

import re
from pathlib import Path

import pytest

BLD_DIR = Path(__file__).parent.parent / "bld"


def parse_example_game_file(filepath: Path) -> list[dict]:
    """Parse an example game markdown file into structured data.

    Args:
        filepath: Path to the example game markdown file.

    Returns:
        List of game dictionaries containing turns with hands and actions.
    """
    content = filepath.read_text()
    games = []

    game_sections = re.split(r"## Example Game", content)[1:]

    for game_section in game_sections:
        game = {"turns": []}
        turn_sections = re.split(r"### Turn \d+:", game_section)[1:]

        for turn_section in turn_sections:
            hand_match = re.search(r"- Hand: \[([\d, ]+)\]", turn_section)
            if not hand_match:
                continue

            hand = [int(x.strip()) for x in hand_match.group(1).split(",")]

            actions = []
            for match in re.finditer(r"\*\*Step \d+:\*\* (.+)", turn_section):
                actions.append(match.group(1))

            game["turns"].append({"hand": hand, "actions": actions})

        if game["turns"]:
            games.append(game)

    return games


def get_example_game_files() -> list[Path]:
    """Get all example game files in the bld directory."""
    return list(BLD_DIR.glob("*_example_games.md"))


@pytest.fixture
def example_game_files() -> list[Path]:
    """Fixture providing all example game files."""
    return get_example_game_files()


def test_example_game_files_exist(example_game_files):
    """Verify that at least one example game file exists."""
    assert len(example_game_files) > 0, "No example game files found in bld/"


@pytest.mark.parametrize("filepath", get_example_game_files(), ids=lambda p: p.stem)
def test_players_only_play_cards_in_hand(filepath: Path):
    """Verify that players only play cards that are in their hand."""
    games = parse_example_game_file(filepath)

    for game_idx, game in enumerate(games):
        for turn_idx, turn in enumerate(game["turns"]):
            hand = set(turn["hand"])

            for action in turn["actions"]:
                if action == "End turn":
                    continue

                play_match = re.match(r"Play (\d+) on", action)
                if play_match:
                    card = int(play_match.group(1))
                    assert card in hand, (
                        f"Game {game_idx + 1}, Turn {turn_idx + 1}: "
                        f"Played card {card} not in hand {sorted(hand)}"
                    )
                    hand.remove(card)


@pytest.mark.parametrize("filepath", get_example_game_files(), ids=lambda p: p.stem)
def test_each_game_has_turns(filepath: Path):
    """Verify that each game has at least one turn."""
    games = parse_example_game_file(filepath)

    for game_idx, game in enumerate(games):
        assert len(game["turns"]) > 0, f"Game {game_idx + 1} has no turns"


@pytest.mark.parametrize("filepath", get_example_game_files(), ids=lambda p: p.stem)
def test_each_turn_has_non_empty_hand(filepath: Path):
    """Verify that each turn starts with a non-empty hand."""
    games = parse_example_game_file(filepath)

    for game_idx, game in enumerate(games):
        for turn_idx, turn in enumerate(game["turns"]):
            assert len(turn["hand"]) > 0, (
                f"Game {game_idx + 1}, Turn {turn_idx + 1}: Empty hand"
            )
