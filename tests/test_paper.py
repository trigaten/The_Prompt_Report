import pytest
from prompt_systematic_review.paperSource import Paper
from datetime import date


def test_paper():
    paper1 = Paper(
        "How to write a paper",
        "Harry Parnasus",
        "example.com",
        date(2000, 2, 2),
        ["keyword1", "keyword2"],
    )
    paper2 = Paper(
        "How to NOT write a paper",
        "John Dickenson",
        "example.com",
        date(2002, 3, 3),
        ["keyword1", "keyword2"],
    )
    alsoPaper1 = Paper(
        "How to write a paper",
        "Dr. Harry Parnasus",
        "https://example2.com",
        date(2000, 2, 5),
        ["keyword1", "keyword2"],
    )

    assert paper1 == alsoPaper1
    assert paper1 != paper2 and paper2 != alsoPaper1
