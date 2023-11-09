from prompt_systematic_review.paperSource import Paper
from prompt_systematic_review import arxiv_source
from multiprocessing import Pool
import pandas as pd
import time
from prompt_systematic_review.download_arxiv_query import queryArchive

"""
Download papers from arxiv and save them to a csv file.
Use the ArXivSource "getPapers" function to download papers.
We achieve parallelism by using the multiprocessing library,
with a pool of NUM_PROCESSES processes (adjust as needed per your machine).
"""


# instantiating the arxiv source
aSource = arxiv_source.ArXivSource()


# downlaod paper function
def downloadPaper(paper: str):
    aSource.getPaperSrc(paper, "./arxivPDFs/")
    # sleep to avoid being blocked
    time.sleep(0.5)


if __name__ == "__main__":
    filename = "arxiv_papers.csv"

    queryArchive(filename)
    df = pd.read_csv(filename)

    papList = []

    NUM_PROCESSES = 16  # adjust as needed per your machine
    with Pool(NUM_PROCESSES) as pool:
        for index, row in df.iterrows():
            papList.append(
                Paper(
                    row["title"],
                    row["firstAuthor"],
                    row["url"],
                    row["dateSubmitted"],
                    row["keywords"],
                )
            )

        pool.map(downloadPaper, papList)
