from prompt_systematic_review import ieee_source
from prompt_systematic_review import arxiv_source
from xml.etree import ElementTree as ET
from prompt_systematic_review.paperSource import Paper
from prompt_systematic_review import semantic_scholar_source
import pandas as pd
from datetime import date
from prompt_systematic_review import keywords
import time

getQuery = True
cleanUpData = True

downloadName = "<FILENAME>.csv"


if getQuery:
    aSource = arxiv_source.ArXivSource()

    papers = []

    for keyWord in keywords.model_list:
        print(f"Getting papers for {keyWord}")
        papers += aSource.getPapers(10000, keyWord)

    print(len(papers))
    titles = [paper.title for paper in papers]
    print("got titles")
    firstAuthors = [paper.firstAuthor for paper in papers]
    urls = [paper.url for paper in papers]
    dateSubmitteds = [paper.dateSubmitted for paper in papers]
    keywordss = [paper.keywords for paper in papers]
    print("got keywords")

    df = pd.DataFrame(
        {
            "title": titles,
            "firstAuthor": firstAuthors,
            "url": urls,
            "dateSubmitted": dateSubmitteds,
            "keywords": keywordss,
        }
    )
    print("DFed")

    df.to_csv(downloadName, index=False)

if cleanUpData:
    df = pd.read_csv(downloadName)

    # removes any duplicate rows based off of column url
    df = df.drop_duplicates(subset=["url"])
    # save df again to csv no index

    df.to_csv(downloadName, index=False)


def i_want_DF_A_and_not_in_DF_B(df_A, df_B):
    return df_A[~df_A.url.isin(df_B.url)]
