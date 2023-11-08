from prompt_systematic_review.paperSource import PaperSource
import requests
from xml.etree import ElementTree as ET
from datetime import date
from typing import List
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
            url = self.baseURL + '"' + keyword + '"' + "&start=0&max_results=" + str(count)
            # Use custom header to avoid being blocked
            print("sentNOW")
            data = requests.get(url, headers=headers).content.decode("utf-8","ignore")
            f = open(f"arxiv_{keyword}_data.xml", "w")
            f.write(data)
            f.close()
            print("gotten data")
            parser = ET.XMLParser(encoding="utf-8")
            root = ET.fromstring(data,parser=parser)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            print("starting arxiv")
            for entry in entries:
                # Extract paper details from entry
                title = entry.find("{http://www.w3.org/2005/Atom}title").text
                firstAuthor = entry.find(
                    "{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name"
                ).text
                url = entry.find("{http://www.w3.org/2005/Atom}id").text.replace("/abs/", "/pdf/")+".pdf"
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

                paper = Paper(
                    title.replace("\n", "").replace("\r", ""),
                    firstAuthor,
                    url,
                    dateSubmitted,
                    [keyWord.lower() for keyWord in keyWords],
                )
                papers.append(paper)
        return papers

    def getPaperSrc(self, paper: Paper, destinationFolder:str, recurse=0 ):
        """
        download a paper.

        :param paper: The paper to get the download of. 
        :type paper: Paper
        :param destinationFolder: The folder to save the paper to.
        :type destinationFolder: str
        :return: nothing
        :rtype: None
        """
        url = paper.url 
        response = requests.get(url)
        #print(response.status_code)
        if (str(response.status_code) != "200" or len(response.content)==0) and recurse < 5:
            time.sleep(2 * recurse)
            self.getPaperSrc(paper, destinationFolder, recurse=recurse+1)
        elif (str(response.status_code) != "200" or len(response.content)==0) and recurse >= 5:
            print("rip")
        else:   
            with open(destinationFolder + url.split("/")[-1], 'wb') as f:
                f.write(response.content)
                print(f"downloaded {str(paper)} with content length {len(response.content)}")
        
        return
    
