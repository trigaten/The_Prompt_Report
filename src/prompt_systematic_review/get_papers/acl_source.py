import requests
from datetime import date, datetime
from typing import List
import os
from prompt_systematic_review.get_papers.paperSource import Paper
from prompt_systematic_review.get_papers.paperSource import PaperSource
from prompt_systematic_review.utils.utils import headers
from acl_anthology import Anthology
import time


class AclSource(PaperSource):
    """A class to represent a source of papers from ArXiv."""

    baseURL = "http://export.arxiv.org/api/query?search_query=all:"
    anthology = Anthology.from_repo()
    papersList = [paper for paper in anthology.papers()]

    def getPapers(self, count: int, keyWords: List[str]) -> List[Paper]:
        """
        Get a list of papers from ACL that match the given keywords.

        :param count: The number of papers to retrieve.
        :type count: int
        :param keyWords: A list of keywords to match.
        :type keyWords: List[str]
        :return: A list of matching papers.
        :rtype: List[Paper]
        """
        papers = []
        for keyword in keyWords:
            for paper in self.papersList:
                if not (
                    paper.deletion is not None
                    or paper.is_deleted
                    or paper.abstract is None
                    or paper.pdf is None
                    or len(paper.authors) == 0
                ):
                    if (
                        keyword
                        in paper.title.as_text().lower()
                        + paper.abstract.as_text().lower()
                    ):
                        if paper.month is not None:
                            if "-" in paper.month:
                                month = paper.month.split("-")[0]
                                s = month + " " + paper.year
                            else:
                                s = paper.month + " " + paper.year
                            try:
                                dateSubmitted = datetime.strptime(s, "%B %Y").date()
                            except ValueError:
                                dateSubmitted = datetime.strptime(s, "%m %Y").date()
                            except Exception as e:
                                print(f"Error processing {paper.title}: {e}")
                        elif paper.year is not None:
                            s = paper.year
                            dateSubmitted = datetime.strptime(s, "%Y").date()
                        else:
                            continue
                        paper = Paper(
                            paper.title.as_text().replace("\n", "").replace("\r", ""),
                            [i.name.as_first_last for i in paper.authors],
                            paper.pdf.url,
                            dateSubmitted,
                            [keyWord.lower() for keyWord in keyWords],
                            paper.abstract.as_text()
                            .replace("\n", "")
                            .replace("\r", ""),
                        )
                        papers.append(paper)
        return papers

    def getPaperSrc(self, paper: Paper, destinationFolder: str = None, recurse=0):
        """
        Download a paper given a paper object.

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
            if destinationFolder:
                with open(destinationFolder + url.split("/")[-1], "wb") as f:
                    f.write(response.content)

        return
