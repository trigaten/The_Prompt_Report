from prompt_systematic_review.paperSource import Paper
from prompt_systematic_review import arxiv_source
from multiprocessing import Pool
import pandas as pd
import time


df = pd.read_csv("<filename>.csv")
# instantiating the arxiv source
aSource = arxiv_source.ArXivSource()


# downlaod paper function
def downloadPaper(paper: str):
    aSource.getPaperSrc(paper, "./arxivPDFs/")
    # sleep to avoid being blocked
    time.sleep(0.5)


if __name__ == "__main__":
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
