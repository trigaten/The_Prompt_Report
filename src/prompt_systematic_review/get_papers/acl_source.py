import requests
from datetime import date, datetime
from typing import List
import os
from prompt_systematic_review.get_papers.paperSource import Paper
from prompt_systematic_review.get_papers.paperSource import PaperSource
from prompt_systematic_review.utils.utils import headers
from anthology import Anthology
import time


class AclSource(PaperSource):
    """A class to represent a source of papers from ArXiv."""

    baseURL = "http://export.arxiv.org/api/query?search_query=all:"
    anthology = Anthology(importdir=os.path.join(os.environ["ACLANTHOLOGY"], "data"))
    papersList = [paper for id_, paper in anthology.papers.items()]

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
                    paper.title is None
                    or paper.is_removed
                    or paper.is_retracted
                    or not paper.has_abstract
                    or not paper.pdf
                ):
                    if keyword in paper.title.lower() + paper.get_abstract().lower():
                        if paper.get("month") is not None:
                            if "-" in paper.get("month"):
                                month = paper.get("month").split("-")[0]
                                s = month + " " + paper.get("year")
                            else:
                                s = paper.get("month") + " " + paper.get("year")
                            try:
                                dateSubmitted = datetime.strptime(s, "%B %Y").date()
                            except ValueError:
                                dateSubmitted = datetime.strptime(s, "%m %Y").date()
                            except Exception as e:
                                print(f"Error processing {paper.title}: {e}")
                        elif paper.get("year") is not None:
                            s = paper.get("year")
                            dateSubmitted = datetime.strptime(s, "%Y").date()
                        else:
                            continue
                        paper = Paper(
                            paper.title.replace("\n", "").replace("\r", ""),
                            paper.get("author_string"),
                            paper.pdf,
                            dateSubmitted,
                            [keyWord.lower() for keyWord in keyWords],
                            paper.get_abstract(),
                        )
                        papers.append(paper)
        return papers

    def getPaperSrc(self, paper: Paper, destinationFolder: str = None, recurse=0):
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
            if destinationFolder:
                with open(destinationFolder + url.split("/")[-1], "wb") as f:
                    f.write(response.content)

        return
