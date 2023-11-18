import requests
from datetime import datetime
from typing import List
from prompt_systematic_review.arxiv_source import ArXivSource
from prompt_systematic_review.paperSource import Paper
import time
from bs4 import BeautifulSoup


class SemanticScholarSource:
    """A class to represent a source of papers from Semantic Scholar."""

    searchBaseURL = "https://api.semanticscholar.org/graph/v1/paper/search"
    paperBaseURL = "https://api.semanticscholar.org/graph/v1/paper/"

    def getPapers(self, count: int, keyWords: List[str]) -> List[Paper]:
        all_papers = []
        max_retries = 5  # Maximum number of retries after hitting rate limit
        for keyword in keyWords:
            query = f'"{keyword}"'  # Query for each keyword
            retry_count = 0  # Count of current retries
            while retry_count < max_retries:
                try:
                    papers_data = self.bulkSearchPapers(query)[
                        :count
                    ]  # Fetch and limit papers for each keyword
                    for paper_data in papers_data:
                        open_access_pdf_url = None
                        if paper_data.get("openAccessPdf"):
                            open_access_pdf_url = paper_data["openAccessPdf"].get("url")

                        paper = Paper(
                            title=paper_data["title"],
                            firstAuthor=paper_data["authors"][0]["name"]
                            if paper_data["authors"]
                            else "",
                            url=open_access_pdf_url,
                            dateSubmitted=datetime.strptime(
                                paper_data["publicationDate"], "%Y-%m-%d"
                            ).date()
                            if paper_data.get("publicationDate")
                            else None,
                            keyWords=[keyword.lower()],  # Single keyword for each paper
                            abstract=paper_data.get("abstract", ""),
                            paperId=paper_data[
                                "paperId"
                            ],  # Adding paperId to Paper object
                        )
                        all_papers.append(paper)
                    break  # Break the loop if successful
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Rate limit error
                        print(f"Rate limit hit for keyword '{keyword}'. Retrying...")
                        retry_count += 1
                        time.sleep(1.1 * retry_count)  # Exponential back-off
                    else:
                        print(f"Error during API request for keyword '{keyword}': {e}")
                        break  # Break the loop on other types of errors
        return all_papers

    def bulkSearchPapers(self, query: str, token: str = None) -> List[dict]:
        bulkSearchURL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
        params = {
            "query": query,
            "fields": "title,authors,abstract,publicationDate,openAccessPdf,paperId",
            "limit": 1000,  # Adjust limit as required
        }
        if token:
            params["token"] = token

        response = requests.get(bulkSearchURL, params=params)
        response.raise_for_status()
        return response.json().get("data", [])

    def getSemanticScholarURL(self, paper: Paper) -> str:
        if not paper.paperId:
            return "Paper ID is not available."

        paper_info_url = (
            f"https://api.semanticscholar.org/graph/v1/paper/{paper.paperId}?fields=url"
        )
        try:
            response = requests.get(paper_info_url)
            response.raise_for_status()
            data = response.json()
            return data.get("url", "URL not found.")
        except requests.exceptions.RequestException as e:
            return f"Request failed: {e}"

    # Not Working Currently
    def getSpecificLinkFromPaper(self, paper: Paper) -> str:
        semantic_scholar_url = self.getSemanticScholarURL(paper)
        print(semantic_scholar_url)
        if not semantic_scholar_url.startswith("http"):
            return "Semantic Scholar URL not found."

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0"
        }

        retries = 3
        delay = 5  # seconds to wait between retries

        while retries > 0:
            response = requests.get(semantic_scholar_url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                specific_div = soup.find("div", class_="alternate-sources__dropdown")
                if specific_div:
                    specific_link = specific_div.find("a", href=True)
                    if specific_link:
                        return specific_link["href"]
                    else:
                        return "Specific link not found in the div."
                else:
                    return "Div with class 'alternate-sources__dropdown' not found."
            elif response.status_code == 202:
                time.sleep(delay)  # Wait for a few seconds before retrying
                retries -= 1
            else:
                return (
                    f"Failed to retrieve the page. Status code: {response.status_code}"
                )

        return "Request accepted but failed to retrieve the page after retries."

    def getPaperSrc(
        self,
        paper: Paper,
        destinationFolder: str,
    ):
        """
        Download a paper if its open access PDF URL is available, using ArxivSource.

        :param paper: The paper to download.
        :type paper: Paper
        :param destinationFolder: The folder to save the paper to.
        :type destinationFolder: str
        """
        if paper.url:
            arxiv_source = ArXivSource()
            arxiv_source.getPaperSrc(paper, destinationFolder)
        else:
            semantic_scholar = SemanticScholarSource()
            paper.url = semantic_scholar.getSpecificLinkFromPaper(paper)
            arxiv_source.getPaperSrc(paper, destinationFolder)


##########################Testing####################################

# Create an instance of the SemanticScholarSource class
semantic_scholar = SemanticScholarSource()

# Define some sample keywords and a count
keywords = ["prompt engineering"]
count = 2  # Number of papers to fetch for each keyword

# Fetch papers
papers = semantic_scholar.getPapers(count, keywords)

# Print the details of each paper
for paper in papers:
    print("Title:", paper.title)
    print("First Author:", paper.firstAuthor)
    print("URL:", paper.url or "Not Available")
    print("Date Submitted:", paper.dateSubmitted)
    print("Keywords:", paper.keywords)
    # print("Abstract:", paper.abstract or "No Abstract Available")
    # Fetch and print the specific link for each paper
    specific_link = semantic_scholar.getSpecificLinkFromPaper(paper)
    print("Specific Link:", specific_link)
    print("-" * 50)  # Separator for readability
