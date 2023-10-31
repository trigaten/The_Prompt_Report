from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from datetime import date
from prompt_systematic_review import keywords
import jellyfish as j


class Paper():

    def __init__(self, title: str, firstAuthor: str, url: str, dateSubmitted: date, keyWords: List[str] ):
        self.title = title
        self.firstAuthor = firstAuthor
        self.url = url
        self.dateSubmitted = dateSubmitted
        self.keywords = keywords
        try:
            assert set(keywords) == set([k.lower() for k in keyWords])
        except:
            raise ValueError("Keywords must be lowercase")
    
    def __str__(self):
        return f"{self.title}, by {self.firstAuthor}".strip()
    
    def __eq__(self, other):
        #this is to handle papers from different sources being the same
        return j.jaro_winkler_similarity(self.__str__().lower() , other.__str__().lower()) > 0.75
                
    
    def matchingKeyWords(self):
        return [keyword for keyword in keywords.keywords_list if keyword in self.keywords]





class PaperSource(ABC):

    baseURL: str

    @abstractmethod
    def getPapers(self, count: int, keyWords: List[str]) -> List[Paper]:
        pass

    @abstractmethod
    def getPaperSrc(self, paper: Paper) -> str:
        pass
