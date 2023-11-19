import requests
from datetime import datetime
from typing import List
from prompt_systematic_review.arxiv_source import ArXivSource
from prompt_systematic_review.paperSource import Paper
import time


class SemanticScholarSource:
    """A class to represent a source of papers from Semantic Scholar."""

    SEARCH_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    PAPER_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/"

    def getPapers(self, count: int, key_words: List[str]) -> List[Paper]:
        """Retrieve a list of papers based on specified keywords.

        Args:
            count (int): Number of papers to retrieve for each keyword.
            key_words (List[str]): List of keywords to search for.

        Returns:
            List[Paper]: A list of Paper objects.
        """
        all_papers = []
        max_retries = 5
        for keyword in key_words:
            query = f'"{keyword}"'
            retry_count = 0
            while retry_count < max_retries:
                try:
                    papers_data = self.bulkSearchPapers(query)[:count]
                    for paper_data in papers_data:
                        if not (
                            paper_data.get("abstract")
                            and paper_data.get("openAccessPdf")
                        ):
                            continue
                        open_access_pdf_url = (
                            paper_data["openAccessPdf"].get("url")
                            if paper_data.get("openAccessPdf")
                            else None
                        )
                        publication_date = (
                            datetime.strptime(
                                paper_data["publicationDate"], "%Y-%m-%d"
                            ).date()
                            if paper_data.get("publicationDate")
                            else None
                        )
                        paper = Paper(
                            title=paper_data["title"],
                            authors=paper_data["authors"]["name"]
                            if paper_data["authors"]
                            else "",
                            url=open_access_pdf_url,
                            dateSubmitted=publication_date,
                            keyWords=None,
                            abstract=paper_data.get("abstract", ""),
                            paperId=paper_data["paperId"],
                        )
                        all_papers.append(paper)
                    break
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code in [429, 504]:
                        print(f"Rate limit hit for keyword '{keyword}'. Retrying...")
                        retry_count += 1
                        time.sleep(1.1 * retry_count)
                    else:
                        print(f"Error during API request for keyword '{keyword}': {e}")
                        break
        return all_papers

    def bulkSearchPapers(self, query: str, token: str = None) -> List[dict]:
        """Perform a bulk search of papers on Semantic Scholar.

        Args:
            query (str): The search query.
            token (str, optional): Token for continuation of a search. Defaults to None.

        Returns:
            List[dict]: A list of paper data in dictionary format.
        """
        bulk_search_url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
        params = {
            "query": query,
            "fields": "title,authors,abstract,publicationDate,openAccessPdf,paperId",
            "limit": 1000,
        }
        if token:
            params["token"] = token

        response = requests.get(bulk_search_url, params=params)
        response.raise_for_status()
        return response.json().get("data", [])

    def getPaperSrc(self, paper: Paper, destination_folder: str):
        """Download a paper if its open access PDF URL is available, using ArXivSource.

        Args:
            paper (Paper): The paper to download.
            destination_folder (str): The folder to save the paper to.
        """
        arxiv_source = ArXivSource()
        if paper.url:
            arxiv_source.getPaperSrc(paper, destination_folder)
