"""Keyterm tokenizing and the trie's scoring; decoding itself runs in the slow Parakeet tests."""

import pytest

from stt_daemon.backends.tdt import Keyterms, tokenize

PIECES = {"▁": 1, "a": 2, "b": 3, "c": 4, "bc": 5, "▁a": 10, "▁ab": 20}


def walk(terms: Keyterms, tokens: list[int]) -> tuple[int, float]:
    node, total = 0, 0.0
    for token in tokens:
        node, change = terms.advance(node, token)
        total += change
    return node, total


def test_a_whole_keyterm_keeps_a_score_per_token() -> None:
    assert walk(Keyterms([(1, 2, 3)], 1.5), [7, 1, 2, 3, 7]) == (0, pytest.approx(4.5))


def test_a_match_that_breaks_off_gives_back_what_it_earned() -> None:
    assert walk(Keyterms([(1, 2, 3)], 1.5), [1, 2, 9]) == (0, pytest.approx(0.0))


def test_a_broken_match_can_start_the_keyterm_again() -> None:
    node, total = walk(Keyterms([(1, 2, 3)], 1.0), [1, 1, 2, 3])
    assert (node, total) == (0, pytest.approx(3.0))


def test_a_match_open_at_the_end_is_counted_back_by_the_caller() -> None:
    terms = Keyterms([(1, 2, 3)], 1.0)
    node, total = walk(terms, [1, 2])
    assert total - terms.open_score(node) == pytest.approx(0.0)


def test_a_keyterm_inside_a_longer_one_keeps_its_score_and_can_go_on() -> None:
    terms = Keyterms([(1, 2), (1, 2, 3)], 1.0)
    node, total = walk(terms, [1, 2])
    assert total - terms.open_score(node) == pytest.approx(2.0)
    assert walk(terms, [1, 2, 3]) == (0, pytest.approx(3.0))
    assert walk(terms, [1, 2, 9]) == (0, pytest.approx(2.0))


def test_a_keyterm_splits_by_the_lowest_id_merge_not_the_longest_piece() -> None:
    assert tokenize(PIECES, "abc") == (10, 5)


def test_a_keyterm_with_a_character_the_model_lacks_is_dropped() -> None:
    assert tokenize(PIECES, "abd") is None
