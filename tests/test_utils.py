# test_utils.py
import pytest
from prompt_systematic_review.utils import search_arxiv, count_articles


def test_search_arxiv():
    data = search_arxiv("covid", max_results=10)
    assert len(data) > 0


def test_count_articles():
    data = search_arxiv("covid", max_results=10)
    assert count_articles(data) == 10
