from prompt_systematic_review.automated_review import review_abstract_title_categorical
import pandas as pd
import os
from dotenv import load_dotenv
import openai
import tqdm
from prompt_systematic_review.utils.utils import process_paper_title
from prompt_systematic_review.config_data import DataFolderPath, DotenvPath

# Load environment variables from the .env file
load_dotenv(dotenv_path=DotenvPath)

# Set the OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")


def evaluate_human_agreement(input_file="arxiv_papers_with_abstract.csv"):
    """
    Evaluate the agreement between AI predictions and human reviews on a dataset.

    This function reads a dataset from a CSV file, processes each paper's title and abstract using the
    `review_abstract_title_categorical` function, and compares the AI predictions with human reviews.

    The results are saved to a new CSV file, and metrics such as precision, recall, accuracy, and F1 score
    are computed and saved to a text file.

    :param input_file: The name of the input CSV file containing the dataset. Defaults to "arxiv_papers_with_abstract.csv".
    :type input_file: str
    """
    # Read input data
    df = pd.read_csv(os.path.join(DataFolderPath, input_file))

    # Initialize an empty list to keep track of results
    results = []

    # Iterate over DataFrame row by row
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        # Apply function to each paper's title and abstract
        result = review_abstract_title_categorical(
            title=row["title"],
            abstract=row["abstract"],
            model="gpt-4-1106-preview",
        )
        # Add result to list
        results.append(result)

    # Add results to DataFrame
    for i, result in enumerate(results):
        df.loc[i, "Probability"] = result["Probability"]
        df.loc[i, "Reasoning"] = result["Reasoning"]

    # Save AI labels to a new CSV file
    df.to_csv(
        os.path.join(
            DataFolderPath,
            "experiments_output" + os.sep + "arxiv_papers_with_ai_labels.csv",
        ),
        index=False,
    )

    # Read blacklist data
    blacklist = pd.read_csv(os.path.join(DataFolderPath, "blacklist.csv"))
    blacklist["title"] = blacklist["title"].apply(lambda x: process_paper_title(x))
    df["title"] = df["title"].apply(lambda x: process_paper_title(x))

    # Filter DataFrame for comparison
    df_limited = df.copy().iloc[200:]
    df_limited["human_review"] = ~df_limited["title"].isin(blacklist["title"])
    keepables = ["highly relevant", "somewhat relevant", "neutral"]

    df_limited["AI_keep"] = df_limited["Probability"].map(
        lambda x: True if x in keepables else False
    )

    # Calculate agreement metrics
    agreement_grid = pd.crosstab(df_limited["AI_keep"], df_limited["human_review"])

    true_positives = agreement_grid.loc[True, True]
    true_negatives = agreement_grid.loc[False, False]
    false_positives = agreement_grid.loc[True, False]
    false_negatives = agreement_grid.loc[False, True]

    accuracy = (true_positives + true_negatives) / len(df_limited)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Print metrics
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1_score}")

    # Write metrics to a text file
    with open(
        os.path.join(
            DataFolderPath, "experiments_output" + os.sep + "agreement_metrics.txt"
        ),
        "w",
    ) as f:
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1 Score: {f1_score}\n")


class Experiment:
    def run():
        evaluate_human_agreement()


if __name__ == "__main__":
    evaluate_human_agreement()
