import requests
from datetime import datetime
from typing import List
from prompt_systematic_review.paperSource import Paper


class SemanticScholarSource:
    """A class to represent a source of papers from Semantic Scholar."""

    baseURL = "http://api.semanticscholar.org/graph/v1/paper/search"

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
                "fields": "title,authors,year",
            }
            response = requests.get(self.baseURL, params=params)
            response.raise_for_status()  # will raise an HTTPError if the HTTP request returned an unsuccessful status code
            data = response.json()

            for paper_data in data["data"]:
                title = paper_data["title"]
                authors = [author["name"] for author in paper_data["authors"]]
                year = paper_data.get("year", None)
                paper_id = paper_data["paperId"]
                url = f"https://api.semanticscholar.org/{paper_id}"

                # Converting year to a date object for consistency with the ArXivSource class
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
