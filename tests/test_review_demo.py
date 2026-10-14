"""Тесты демо-утилит (src/review_demo.py)."""

from __future__ import annotations

import unittest

from src.review_demo import (
    average_score,
    merge_history,
    normalize_tags,
    top_tags,
)


class ReviewDemoTest(unittest.TestCase):

    def test_normalize_tags(self) -> None:
        self.assertEqual(normalize_tags("api, UI;;  api , db"),
                         ["api", "UI", "db"])

    def test_normalize_tags_empty(self) -> None:
        self.assertEqual(normalize_tags("  ,;;"), [])

    def test_average_score(self) -> None:
        self.assertEqual(average_score([5, 4, 3]), 4.0)
        self.assertEqual(average_score([1, 2]), 1.5)

    def test_top_tags(self) -> None:
        counts = [("ui", 2), ("api", 5), ("db", 5), ("misc", 1)]
        self.assertEqual(top_tags(counts, 2),
                         [("api", 5), ("db", 5)])

    def test_merge_history(self) -> None:
        history = ["a"]
        result = merge_history(["b", "c"], history)
        self.assertEqual(result, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
