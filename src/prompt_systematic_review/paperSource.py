from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from datetime import date
from prompt_systematic_review import keywords
from prompt_systematic_review import utils


class Paper:
    def __init__(
        self,
        title: str,
        authors: List[str],
        url: str,
        dateSubmitted: date,
        keyWords: List[str],
        abstract: str,
        paperId: str = None,
    ):
        self.title = title
        self.authors = authors
        self.url = url
        self.dateSubmitted = dateSubmitted
        self.keywords = keyWords
        self.abstract = abstract
        self.paperId = paperId

        try:
            assert set(keyWords) == set([k.lower() for k in keyWords])
        except:
            raise ValueError("Keywords must be lowercase")

    def __str__(self):
        return f"{self.title}, by {self.firstAuthor}".strip()

    def __eq__(self, other):
        # this is to handle papers from different sources being the same
        return utils.process_paper_title(self.title) == utils.process_paper_title(
            other.title
        )

    def __hash__(self):
        return hash(utils.process_paper_title(self.title))

    def matchingKeyWords(self):
        return [
            keyword for keyword in keywords.keywords_list if keyword in self.keywords
        ]

    def to_dict(self):
        """Serialize the paper object to a dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "url": self.url,
            "dateSubmitted": self.dateSubmitted.isoformat()
            if self.dateSubmitted
            else None,
            "keyWords": self.keywords,
            "abstract": self.abstract,
            "paperId": self.paperId,
        }


class PaperSource(ABC):
    baseURL: str

    @abstractmethod
    def getPapers(self, count: int, keyWords: List[str]) -> List[Paper]:
        pass

    @abstractmethod
    def getPaperSrc(self, paper: Paper) -> str:
        pass
