from prompt_systematic_review.paperSource import Paper
from prompt_systematic_review import arxiv_source
import tqdm
import requests
from multiprocessing import Pool
import pandas as pd
import time


df = pd.read_csv("<filename>.csv")
print(df.head())
aSource = arxiv_source.ArXivSource()


def downloadPaper(paper: str):
    aSource.getPaperSrc(paper, "./<destinationFolder>/")
    time.sleep(0.5)
    print("downloaded " + str(paper))


if __name__ == "__main__":
    papList = []

    with Pool(16) as pool:
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
