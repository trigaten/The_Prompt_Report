from prompt_systematic_review.paperSource import Paper
from prompt_systematic_review import arxiv_source
from multiprocessing import Pool
import pandas as pd
import time
from prompt_systematic_review.download_arxiv_query import query_archive
import os
from tqdm import tqdm
from tika import parser

"""
Download papers from arxiv and save them to a csv file.
Use the ArXivSource "getPapers" function to download papers.
We achieve parallelism by using the multiprocessing library,
with a pool of NUM_PROCESSES processes (adjust as needed per your machine).

Then, remove papers from arxiv_papers.csv and from arxivPDFs that don't contain the word prompt
"""

# instantiating the arxiv source
aSource = arxiv_source.ArXivSource()


# downlaod paper function
def downloadPaper(paper: str):
    aSource.getPaperSrc(paper, "./arxivPDFs/")
    # sleep to avoid being blocked
    time.sleep(0.5)


def filter_and_save_pdfs(folder_path, csv_path, output_csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Add a new column "pdf_titles" by extracting titles from the URLs
    df["pdf_titles"] = df["url"].apply(lambda x: os.path.basename(x))

    kept_pdfs = []

    # Iterate through all files in the folder
    for filename in tqdm(os.listdir(folder_path)):
        try:
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)

                parsed_pdf = parser.from_file(file_path)
                data = parsed_pdf["content"]

                if "prompt" in data.lower():
                    kept_pdfs.append(filename)
                else:
                    os.remove(file_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Create a new DataFrame with entries for the kept PDFs
    kept_df = df[df["pdf_titles"].isin(kept_pdfs)]

    # Save the new DataFrame as a new CSV
    kept_df.to_csv(output_csv_path, index=False)

    return len(kept_pdfs)


if __name__ == "__main__":
    filename = "arxiv_papers.csv"
    # mkdir "./arxivPDFs/" if it doesn't exist
    if not os.path.exists("./arxivPDFs/"):
        os.mkdir("./arxivPDFs/")

    # download paper info to csv
    query_archive(filename)
    df = pd.read_csv(filename)

    papList = []

    NUM_PROCESSES = 12  # adjust as needed per your machine
    with Pool(NUM_PROCESSES) as pool:
        for index, row in df.iterrows():
            papList.append(
                Paper(
                    row["title"],
                    row["authors"],
                    row["url"],
                    row["dateSubmitted"],
                    row["keywords"],
                )
            )

        # download pdf of paper
        pool.map(downloadPaper, papList)

    # Provide the path to the folder containing PDFs, the original CSV file, and the output CSV file
    folder_path = "arxivPDFs"
    csv_path = "arxiv_papers.csv"
    output_csv_path = "filtered_arxiv_papers.csv"

    result = filter_and_save_pdfs(folder_path, csv_path, output_csv_path)

    print(f"Number of PDFs containing the word 'prompt' and saved to new CSV: {result}")
