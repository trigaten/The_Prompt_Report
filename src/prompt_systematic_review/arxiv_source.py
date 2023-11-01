from prompt_systematic_review.paperSource import PaperSource
import requests
from xml.etree import ElementTree as ET
from datetime import date
from typing import List
from prompt_systematic_review.paperSource import Paper
from prompt_systematic_review.paperSource import PaperSource
from prompt_systematic_review.utils import headers

class ArXivSource(PaperSource):
    """A class to represent a source of papers from ArXiv."""

    baseURL = "http://export.arxiv.org/api/query?search_query=all:"

    def getPapers(self, count: int, keyWords: List[str]) -> List[Paper]:
        """
        Get a list of papers from ArXiv that match the given keywords.

        :param count: The number of papers to retrieve.
        :type count: int
        :param keyWords: A list of keywords to match.
        :type keyWords: List[str]
        :return: A list of matching papers.
        :rtype: List[Paper]
        """
        papers = []
        for keyword in keyWords:
            url = self.baseURL + keyword + "&start=0&max_results=" + str(count)
            # Use custom header to avoid being blocked
            data = requests.get(url, headers=headers).content
            root = ET.fromstring(data)
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            for entry in entries:
                # Extract paper details from entry
                title = entry.find('{http://www.w3.org/2005/Atom}title').text
                firstAuthor = entry.find('{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name').text
                url = entry.find('{http://www.w3.org/2005/Atom}id').text
                dateSubmitted = entry.find('{http://www.w3.org/2005/Atom}published').text
                dateSubmitted = date(int(dateSubmitted[:4]), int(dateSubmitted[5:7]), int(dateSubmitted[8:10]))
                keyWords = [keyword.attrib['term'] for keyword in entry.findall('{http://www.w3.org/2005/Atom}category')]
                
                paper = Paper(title, firstAuthor, url, dateSubmitted, [keyword.lower() for keyWord in keyWords])
                papers.append(paper)
        return papers

    def getPaperSrc(self, paper: Paper) -> str:
        """
        Get the source of a paper.

        :param paper: The paper to get the source of.
        :type paper: Paper
        :return: The source of the paper.
        :rtype: str
        """
        url = paper.url + ".pdf"
        response = requests.get(url)
        return response.content.decode('utf-8')