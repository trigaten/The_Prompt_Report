import os
import json
from prompt_systematic_review.get_papers.semantic_scholar_source import (
    SemanticScholarSource,
)
from prompt_systematic_review.utils.keywords import keywords_list
import pandas as pd
from tqdm import tqdm


def create_directory(directory_name):
    """
    Create a directory if it doesn't already exist.

    :param directory_name: The name of the directory to create.
    :type directory_name: str
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def save_papers_to_json(papers, file_path):
    """
    Save a list of Paper objects to a JSON file.

    :param papers: A list of Paper objects to save.
    :type papers: List[Paper]
    :param file_path: The file path to save the JSON file.
    :type file_path: str
    """
    papers_dict = [paper.to_dict() for paper in papers]
    with open(file_path, "w") as file:
        json.dump(papers_dict, file, indent=4)


def query_semantic_scholar(downloadName: str = None, verbose=False):
    """
    Query Semantic Scholar for papers based on a list of keywords and save the results to a CSV file.

    This function uses the SemanticScholarSource class to retrieve papers that match the specified keywords.
    It iterates over the list of keywords and retrieves papers for each keyword using the getPapers method.
    The retrieved papers are then combined into a single DataFrame.

    If the `downloadName` parameter is provided, the DataFrame is saved to a CSV file with the specified name.
    The function also returns the combined DataFrame.

    :param downloadName: The name of the CSV file to save the papers to (optional).
    :type downloadName: str
    :param verbose: Whether to display progress information using tqdm (default is False).
    :type verbose: bool
    :return: A DataFrame containing the retrieved papers.
    :rtype: pd.DataFrame
    """
    sss = SemanticScholarSource()

    all_papers_df = pd.DataFrame()

    if verbose:
        iterator = tqdm(keywords_list, desc="Processing keywords")
    else:
        iterator = keywords_list

    for keyword in iterator:
        papers = sss.getPapers(300, [keyword])
        papers_data = [paper.to_dict() for paper in papers]
        papers_df = pd.DataFrame(papers_data)
        all_papers_df = pd.concat([all_papers_df, papers_df], ignore_index=True)

    if downloadName:
        all_papers_df.to_csv(downloadName, index=False)
        if verbose:
            print(f"Saved all papers to '{downloadName}'.")

    return all_papers_df
