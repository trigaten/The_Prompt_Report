from prompt_systematic_review.get_papers.paperSource import Paper
from prompt_systematic_review.get_papers.arxiv_source import ArXivSource
from prompt_systematic_review.get_papers.semantic_scholar_source import (
    SemanticScholarSource,
)
from datetime import date, datetime
from prompt_systematic_review.utils.utils import process_paper_title
import pytest
from prompt_systematic_review.download_arxiv_query import query_archive


def test_paper():
    paper1 = Paper(
        "How to write a paper",
        "Harry Parnasus",
        "example.com",
        date(2000, 2, 2),
        ["keyword1", "keyword2"],
        "amongus babongus",
    )
    paper2 = Paper(
        "How to NOT write a paper",
        "John Dickenson",
        "example.com",
        date(2002, 3, 3),
        ["keyword1", "keyword2"],
        "amongus babongus",
    )
    alsoPaper1 = Paper(
        "How to write a paper",
        "Dr. Harry Parnasus",
        "https://example2.com",
        date(2000, 2, 5),
        ["keyword1", "keyword2"],
        "amongus babongus",
    )

    assert paper1 == alsoPaper1
    assert paper1 != paper2 and paper2 != alsoPaper1


# mark as API_test, so it can be excluded by CI pipeline
@pytest.mark.API_test
def test_arxiv_source():
    # test that arXiv source returns papers properly
    arxiv_source = ArXivSource()
    papers = arxiv_source.getPapers(2, ["machine learning"])
    assert len(papers) == 2
    for paper in papers:
        assert isinstance(paper, Paper)
        # assert "machine learning" in paper.keywords
        # paper_src = arxiv_source.getPaperSrc(paper)
        # assert isinstance(paper_src, str)
        # assert len(paper_src) > 0

    # test that arXiv source returns the exact information for one paper properly
    arxiv_source = ArXivSource()
    TITLE = "Foundational Models in Medical Imaging: A Comprehensive Survey and Future Vision"
    papers = arxiv_source.getPapers(1, [TITLE])
    paper = papers[0]
    assert process_paper_title(paper.title) == process_paper_title(TITLE.lower())
    assert "Bobby Azad" in paper.authors
    assert paper.url == "http://arxiv.org/pdf/2310.18689v1.pdf"

    date_string = "Sat, 28 Oct 2023 12:08:12 UTC"
    date_object = datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S %Z").date()
    assert paper.dateSubmitted == date_object
    # assert paper.keywords == [
    #     "foundational models in medical imaging: a comprehensive survey and future vision"
    # ]


@pytest.mark.API_test
def test_semantic_scholar_get_papers():
    semantic_scholar = SemanticScholarSource()
    keywords_list = [["jailbreak prompt"], ["prompt an llm"]]
    count = 5
    papers = semantic_scholar.getPapers(count, keywords_list)

    for paper in papers:
        assert paper.title, "Paper must have a title"
        assert paper.url, "Paper must have a URL"
        assert isinstance(
            paper.dateSubmitted, date
        ), "Paper must have a submission date of type date"
        assert paper.authors, "Paper must have at least one author"
        assert paper.abstract, "Paper must have an abstract"


def test_query_archive():
    # Makes sure that all the proper headings for all the data are in the csv file

    response_file = query_archive("querycsv")
    assert "title" in response_file
    assert "authors" in response_file
    assert "url" in response_file
    assert "dateSubmitted" in response_file
    assert "keywords" in response_file
    assert "abstract" in response_file
