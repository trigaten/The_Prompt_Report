from prompt_systematic_review.get_papers.download_arxiv import query_arxiv
from prompt_systematic_review.get_papers.download_semantic_scholar import (
    query_semantic_scholar,
)
from prompt_systematic_review.get_papers.download_acl import query_acl
from prompt_systematic_review.automated_review import review_abstract_title_categorical
from prompt_systematic_review.config_data import DataFolderPath, DotenvPath

import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import PyPDF2
from PyPDF2.errors import PdfReadError
from prompt_systematic_review.utils.utils import process_paper_title

import openai
import tqdm
from dotenv import load_dotenv
import logging

# don't want to see warning messages when users are running
pdflogger = logging.getLogger("PyPDF2")
pdflogger.setLevel(logging.ERROR)
urlLogger = logging.getLogger("urllib3")
urlLogger.setLevel(logging.ERROR)

load_dotenv(dotenv_path=DotenvPath)  # load all entries from .env file

openai.api_key = os.getenv("OPENAI_API_KEY")


def downloadPaper(url: str, title: str):
    """
    Download a paper given its URL and title.

    :param url: The URL of the paper to download.
    :type url: str
    :param title: The title of the paper.
    :type title: str
    """
    response = requests.get(url)
    recurse = 0
    while (
        str(response.status_code) != "200" or len(response.content) == 0
    ) and recurse < 5:
        # if failed to download try again after waiting 2*recurse seconds
        time.sleep(2 * recurse)
        response = requests.get(url)
        recurse += 1

    if str(response.status_code) == "200" and len(response.content) != 0:
        # replace invalid characters in title
        title = process_paper_title(title=title)
        name = title + ".pdf"
        data_folder_path = os.path.join(DataFolderPath, "papers")
        with open(os.path.join(data_folder_path, name), "wb") as f:
            f.write(response.content)


def collect():
    """
    Collect papers from various sources, deduplicate and filter them, and save them to a CSV file.

    This function performs the following steps:
    1. Downloads papers from arXiv, Semantic Scholar, and ACL using the respective query functions.
    2. Cleans and deduplicates the downloaded papers.
    3. Removes papers that are in the blacklist.
    4. Downloads the PDF files of the remaining papers using multithreading.
    5. Filters out papers that don't contain the word "prompt" in their content.
    6. Performs an automated review of the papers using the GPT-4 model.
    7. Combines the human-reviewed and AI-reviewed papers into a final dataset.
    8. Removes PDF files of papers that are not in the final dataset.
    9. Saves the final dataset to a CSV file named "master_papers.csv".
    """
    # download CSV of arXiv results
    arxiv_df = query_arxiv(verbose=True)
    # clean arXiv CSV
    arxiv_df["title"] = arxiv_df["title"].apply(lambda x: process_paper_title(x))
    arxiv_df["source"] = "arXiv"

    semantic_scholar_df = query_semantic_scholar(verbose=True)
    # clean Semantic CSV
    semantic_scholar_df["title"] = semantic_scholar_df["title"].apply(
        lambda x: process_paper_title(x)
    )
    semantic_scholar_df["source"] = "Semantic Scholar"

    # download ACL CSV
    acl_df = query_acl(verbose=True)
    # clean ACL CSV
    acl_df["title"] = acl_df["title"].apply(lambda x: process_paper_title(x))
    acl_df["source"] = "ACL"

    # combine dfs
    combined_df = pd.concat([semantic_scholar_df, arxiv_df, acl_df])
    # drop duplicates
    deduplicated_df = combined_df.drop_duplicates(subset="title")

    blacklist = pd.read_csv(os.path.join(DataFolderPath, "blacklist.csv"))
    blacklist["title"] = blacklist["title"].apply(lambda x: process_paper_title(x))
    deduplicated_df = deduplicated_df[
        ~deduplicated_df["title"].isin(blacklist["title"])
    ]

    data = list(zip(deduplicated_df["url"].tolist(), deduplicated_df["title"].tolist()))

    # make papers folder if it doesn't already exist
    os.makedirs(os.path.join(DataFolderPath, "papers"), exist_ok=True)

    NUM_PROCESSES = 12  # adjust as needed per your machine
    with ThreadPoolExecutor(max_workers=NUM_PROCESSES) as executor:
        executor.map(lambda p: downloadPaper(*p), data)

    new_blacklist = []

    # Iterate over the files in the directory
    for filename in tqdm.tqdm(os.listdir(os.path.join(DataFolderPath, "papers"))):
        try:
            if filename.endswith(".pdf"):
                file_path = os.path.join(
                    DataFolderPath, os.path.join("papers", filename)
                )
                with open(file_path, "rb") as file:
                    pdf = PyPDF2.PdfReader(file)
                    contains_prompt = False
                    for page in pdf.pages:
                        if "prompt" in page.extract_text().lower():
                            contains_prompt = True
                            break

                if not contains_prompt:
                    # Delete the file
                    os.remove(file_path)
                    # Drop the corresponding row from the dataframe
                    deduplicated_df = deduplicated_df[
                        deduplicated_df["title"] != filename[:-4]
                    ]
                    # Add the paper to the new blacklist
                    new_blacklist += filename[:-4]

        except Exception as e:
            # Delete the file if cant be read
            os.remove(file_path)
            # Drop the corresponding row from the dataframe
            deduplicated_df = deduplicated_df[deduplicated_df["title"] != filename[:-4]]
            # PDFRead Error is likely because of corrupted or empty PDF, can be ignored
            if str(e) != "EOF marker not found":
                print(f"Error processing {filename}: {e}")
    

    # Get a list of all the paper titles in the directory (without the .pdf extension)
    paper_titles = [
        filename[:-4]
        for filename in os.listdir(os.path.join(DataFolderPath, "papers"))
        if filename.endswith(".pdf")
    ]

    # Remove any rows from deduplicated_df where the title is not in paper_titles
    deduplicated_df = deduplicated_df[deduplicated_df["title"].isin(paper_titles)]
    # Load the csv file
    df_for_review = pd.read_csv(
        os.path.join(DataFolderPath, "arxiv_papers_for_human_review.csv")
    )

    df_for_review["title"] = df_for_review["title"].apply(
        lambda x: process_paper_title(x)
    )
    # Get a list of the titles in the csv file

    titles_for_review = df_for_review["title"].tolist()

    # have been human reviewed as correct
    df_safe = deduplicated_df[deduplicated_df["title"].isin(titles_for_review)]
    # need ai review
    df_for_ai_review = deduplicated_df[
        ~deduplicated_df["title"].isin(titles_for_review)
    ]

    results = []

    # Iterate over DataFrame row by row
    for index, row in tqdm.tqdm(df_for_ai_review.iterrows()):
        # Apply function to each paper's title and abstract
        result = review_abstract_title_categorical(
            title=row["title"],
            abstract=row["abstract"],
            model="gpt-4-1106-preview",
        )
        # Add result to list
        results.append(result)

    for i, result in enumerate(results):
        df_for_ai_review.loc[i, "Probability"] = result["Probability"]
        df_for_ai_review.loc[i, "Reasoning"] = result["Reasoning"]

    keepables = ["highly relevant", "somewhat relevant", "neutral"]
    others = ["somewhat irrelevant", "highly irrelevant"]

    df_ai_reviewed_positive = df_for_ai_review[
        df_for_ai_review["Probability"].isin(keepables)
    ]
    df_ai_reviewed_negative = df_for_ai_review[
        df_for_ai_review["Probability"].isin(others)
    ]
    df_combined = pd.concat([df_safe, df_ai_reviewed_positive], ignore_index=True)
    paper_titles = [
        filename[:-4]
        for filename in os.listdir(os.path.join(DataFolderPath, "papers"))
        if filename.endswith(".pdf")
    ]

    # Remove any rows from deduplicated_df where the title is not in paper_titles
    df_combined = df_combined[df_combined["title"].isin(paper_titles)]
    # Get a list of all titles in df_combined
    df_titles = df_combined["title"].tolist()
    c = 0
    # Iterate over all files in the "papers" directory
    for filename in os.listdir(os.path.join(DataFolderPath, "papers")):
        # Check if the file is a PDF and its title is not in df_titles
        if filename.endswith(".pdf") and filename[:-4] not in df_titles:
            # Remove the file
            os.remove(DataFolderPath + os.sep + "papers" + os.sep + filename)
    df_combined.to_csv(os.path.join(DataFolderPath, "master_papers.csv"))
