from prompt_systematic_review import arxiv_source
from prompt_systematic_review.paperSource import Paper
import pandas as pd
from datetime import date
from prompt_systematic_review import keywords
import tqdm


def query_archive(downloadName: str = None, verbose=False):
    """
    Download papers from arxiv and save them to a csv file.
    :param downloadName: The name of the csv file to save the data to.
    """

    aSource = arxiv_source.ArXivSource()

    papers = []
    if verbose:
        for keyWord in tqdm.tqdm(keywords.keywords_list):
            # go through keywords list and download
            papers += aSource.getPapers(10000, keyWord)
    else:
        for keyWord in keywords.keywords_list:
            # go through keywords list and download
            papers += aSource.getPapers(10000, keyWord)

    # make dataframe
    titles = [paper.title for paper in papers]
    authors = [paper.authors for paper in papers]
    urls = [paper.url for paper in papers]
    dateSubmitteds = [paper.dateSubmitted for paper in papers]
    keywordss = [paper.keywords for paper in papers]
    abstracts = [paper.abstract for paper in papers]

    df = pd.DataFrame(
        {
            "title": titles,
            "authors": authors,
            "url": urls,
            "dateSubmitted": dateSubmitteds,
            "keywords": keywordss,
            "abstract": abstracts,
        }
    )
    # drop duplicates
    df = df.drop_duplicates(subset=["url"])

    # optional save to file
    if downloadName:
        df.to_csv(downloadName, index=False)

    return df
