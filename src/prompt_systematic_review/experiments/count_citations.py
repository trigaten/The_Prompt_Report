import requests
import pandas as pd
from collections import Counter
from prompt_systematic_review.config_data import DataFolderPath
import os

file_path = os.path.join(DataFolderPath, "master_papers.csv")

arxiv_papers_df = pd.read_csv(file_path)

paper_ids = arxiv_papers_df["paperId"]

citation_counts_id = Counter(paper_ids)
citation_counts_title = Counter()


def count_citations(paper_name: str) -> int:
    """Count the number of citations for a given paper name"""

    citation_counts = 0

    for paper_id in paper_ids:
        # Fetch the citation info for given paper_id
        response = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields=paperId,title"
        )
        try:
            data_lst = response.json()["data"]
            for data in data_lst:
                title = data["citingPaper"]["title"]
                if title == paper_name:
                    citation_counts_id[paper_id] += 1
                    citation_counts_title[title] += 1
        except Exception as e:
            print(e)

        # return number of paper citations
        return citation_counts


class Experiment:
    def run():
        count_citations()
