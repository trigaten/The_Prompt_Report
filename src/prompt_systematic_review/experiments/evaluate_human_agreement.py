from prompt_systematic_review.automated_review import review_abstract_title_categorical
import pandas as pd
import os
from dotenv import load_dotenv
import openai
import tqdm
from prompt_systematic_review.utils.utils import process_paper_title

from prompt_systematic_review.config_data import DataFolderPath, DotenvPath

load_dotenv(dotenv_path=DotenvPath)  # load all entries from .env file

openai.api_key = os.getenv("OPENAI_API_KEY")


def evaluate_human_agreement(inputFile="arxiv_papers_with_abstract.csv"):
    df = pd.read_csv(os.path.join(DataFolderPath, inputFile))
    # Empty list to keep track of results
    results = []

    # Iterate over DataFrame row by row
    for index, row in tqdm.tqdm(df.iterrows()):
        # Apply function to each paper's title and abstract
        result = review_abstract_title_categorical(
            title=row["title"],
            abstract=row["abstract"],
            model="gpt-4-1106-preview",
        )
        # Add result to list
        results.append(result)

    for i, result in enumerate(results):
        df.loc[i, "Probability"] = result["Probability"]
        df.loc[i, "Reasoning"] = result["Reasoning"]

    df.to_csv(
        os.path.join(
            DataFolderPath,
            "experiments_output" + os.sep + "arxiv_papers_with_ai_labels.csv",
        )
    )
    blacklist = pd.read_csv(os.path.join(DataFolderPath, "blacklist.csv"))
    blacklist["title"] = blacklist["title"].apply(lambda x: process_paper_title(x))
    df["title"] = df["title"].apply(lambda x: process_paper_title(x))

    # df = df.iloc[400:800]
    df_limited = df.copy().iloc[200:]
    df_limited["human_review"] = ~df_limited["title"].isin(blacklist["title"])
    keepables = ["highly relevant", "somewhat relevant", "neutral"]

    df_limited["AI_keep"] = df_limited["Probability"].map(
        lambda x: True if x in keepables else False
    )
    num_same_rows = (df_limited["AI_keep"] == df_limited["human_review"]).sum()
    num_same_rows / len(df_limited["human_review"])

    agreement_grid = pd.crosstab(df_limited["AI_keep"], df_limited["human_review"])

    true_positives = agreement_grid.loc[True, True]
    true_negatives = agreement_grid.loc[False, False]
    false_positives = agreement_grid.loc[True, False]
    false_negatives = agreement_grid.loc[False, True]

    accuracy = (true_positives + true_negatives) / len(df_limited)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1_score}")


class Experiment:
    def run():
        evaluate_human_agreement()
