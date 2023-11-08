import requests
from datetime import datetime
from typing import List
from prompt_systematic_review.paperSource import Paper


class SemanticScholarSource:
    """A class to represent a source of papers from Semantic Scholar."""

    searchBaseURL = "https://api.semanticscholar.org/graph/v1/paper/search"
    paperBaseURL = "https://api.semanticscholar.org/graph/v1/paper/"

    def getPapers(
        self, keyWords: List[str], count: int = 10, offset: int = 0
    ) -> List[Paper]:
        """
        Get a list of papers from Semantic Scholar that match the given keywords.
        :param keyWords: A list of keywords to match.
        :param count: The number of papers to retrieve.
        :param offset: The offset to start retrieving papers from.
        :return: A list of matching papers.
        """
        papers = []
        for keyword in keyWords:
            params = {
                "query": keyword,
                "offset": offset,
                "limit": count,
                "fields": "title,authors,year,paperId",
            }
            response = requests.get(self.searchBaseURL, params=params)
            response.raise_for_status()
            data = response.json()

            for paper_data in data.get("data", []):
                title = paper_data.get("title")
                authors = [
                    author.get("name") for author in paper_data.get("authors", [])
                ]
                year = paper_data.get("year", None)
                paper_id = paper_data.get("paperId")
                url = f"https://api.semanticscholar.org/{paper_id}"

                date_submitted = datetime(year, 1, 1).date() if year else None

                paper = Paper(
                    title=title,
                    firstAuthor=authors[0] if authors else "",
                    url=url,
                    dateSubmitted=date_submitted,
                    keyWords=[keyword.lower()],
                )
                papers.append(paper)
        return papers

    def getPaperDetails(self, paperId: str) -> dict:
        """Get the detailed information of a paper from Semantic Scholar."""
        url = f"{self.paperBaseURL}{paperId}?fields=url,openAccessPdf"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data

    def getPaperPDF(self, paperId: str) -> str:
        """Get the URL of a paper from Semantic Scholar."""
        paper_details = self.getPaperDetails(paperId)
        open_access_pdf_data = paper_details.get("openAccessPdf")
        if open_access_pdf_data:
            return open_access_pdf_data.get("url")
        return None

    def getOpenAccessPapers(self, keyWords: List[str], n: int) -> List[Paper]:
        """
        Get a list of n papers with an open access PDF from Semantic Scholar that match the given keywords.
        :param keyWords: A list of keywords to match.
        :param n: The number of papers with an open access PDF to retrieve.
        :return: A list of papers with an open access PDF.
        """
        open_access_papers = []
        offset = 0
        while len(open_access_papers) < n:
            batch = self.getPapers(keyWords, count=100, offset=offset)
            for paper in batch:
                if len(open_access_papers) >= n:
                    break
                if self.getPaperPDF(paper.url.split("/")[-1]):
                    open_access_papers.append(paper)
            offset += len(batch)
        return open_access_papers
