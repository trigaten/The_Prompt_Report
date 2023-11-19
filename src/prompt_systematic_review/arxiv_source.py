from prompt_systematic_review.paperSource import PaperSource
import requests
from xml.etree import ElementTree as ET
from datetime import date
from typing import List
import os
from prompt_systematic_review.paperSource import Paper
from prompt_systematic_review.paperSource import PaperSource
from prompt_systematic_review.utils import headers
import time


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
            url = (
                self.baseURL
                + '"'
                + keyword
                + '"'
                + "&start=0&max_results="
                + str(count)
            )
            # Use custom header to avoid being blocked
            data = requests.get(url, headers=headers).content.decode("utf-8", "ignore")
            # create "./.log/" folder if it doesn't exist
            if not os.path.exists("./.log/"):
                os.mkdir("./.log/")
            f = open(f"./.log/arxiv_{keyword}_data.xml", "w", encoding="utf-8")
            f.write(data)
            f.close()
            parser = ET.XMLParser(encoding="utf-8")
            root = ET.fromstring(data, parser=parser)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            for entry in entries:
                # Extract paper details from entry
                title = entry.find("{http://www.w3.org/2005/Atom}title").text
                authors = [
                    author.find("{http://www.w3.org/2005/Atom}name").text
                    for author in entry.findall("{http://www.w3.org/2005/Atom}author")
                ]
                url = (
                    entry.find("{http://www.w3.org/2005/Atom}id").text.replace(
                        "/abs/", "/pdf/"
                    )
                    + ".pdf"
                )
                dateSubmitted = entry.find(
                    "{http://www.w3.org/2005/Atom}published"
                ).text
                dateSubmitted = date(
                    int(dateSubmitted[:4]),
                    int(dateSubmitted[5:7]),
                    int(dateSubmitted[8:10]),
                )
                keyWords = [
                    keyword.attrib["term"]
                    for keyword in entry.findall(
                        "{http://www.w3.org/2005/Atom}category"
                    )
                ]
                abstract = (
                    entry.find("{http://www.w3.org/2005/Atom}summary")
                    .text.replace("\n", "")
                    .replace("\r", "")
                )

                paper = Paper(
                    title.replace("\n", "").replace("\r", ""),
                    authors,
                    url,
                    dateSubmitted,
                    [keyWord.lower() for keyWord in keyWords],
                    abstract,
                )
                papers.append(paper)
        return papers

    def getPaperSrc(self, paper: Paper, destinationFolder: str, recurse=0):
        """
        download a paper.

        :param paper: The paper to get the download of.
        :type paper: Paper
        :param destinationFolder: The folder to save the paper to.
        :type destinationFolder: str
        :param recurse: hidden recursion parameter (repeat download attempt if fail), max recursion depth is 5
        :type recurse: int
        :return: nothing
        :rtype: None
        """
        url = paper.url
        response = requests.get(url)
        if (
            str(response.status_code) != "200" or len(response.content) == 0
        ) and recurse < 5:
            # if failed to download try again after waiting 2*recurse seconds
            time.sleep(2 * recurse)
            self.getPaperSrc(paper, destinationFolder, recurse=recurse + 1)
        elif (
            str(response.status_code) != "200" or len(response.content) == 0
        ) and recurse >= 5:
            # if failed to download after 5 attempts, give up
            pass
        else:
            with open(destinationFolder + url.split("/")[-1], "wb") as f:
                f.write(response.content)

        return
