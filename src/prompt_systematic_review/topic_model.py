import subprocess
import os
import pandas as pd
from importlib import resources as rsc


"""
Runs the topic model with default parameters seen in data/topic-model-data/README.md (MANUAL section). 
Results are found in topic-model-data/topic-model-outputs folder
Runs all topic models (10, 25, 50). If you don't want to run all of them, feel free to comment out below.

"""

"""Summary: Changes directory to /data
"""
def change_dir_data():
    DIRNAME = os.path.split(__file__)[0]
    BACK = os.sep + os.pardir
    os.chdir(os.path.normpath(DIRNAME + BACK + BACK))
    os.chdir(os.path.join(os.getcwd(), "data"))

"""Summary: Changes directory to /src/prompt_systematic_review
"""
def change_dir_src():
    DIRNAME = os.path.split(__file__)[0]
    os.chdir(DIRNAME)


"""Summary: Downloads master_papers.csv from PromptSystematicReview huggingface database to topic-model folder
"""


def get_csv():
    df = pd.read_csv(
        "https://huggingface.co/datasets/PromptSystematicReview/Prompt_Systematic_Review_Dataset/raw/main/master_papers.csv"
    )
    df.to_csv(
        "./topic-model-data/master_papers.csv",
        sep=",",
        index=False,
        encoding="utf-8",
    )


"""Summary: Creates detected-phrases folder and calls detect phrases from soup-nuts
"""


def run_detect_phrases():
    os.mkdir("./topic-model-data/detected-phrases")

    subprocess.run(
        [
            "soup-nuts",
            "detect-phrases",
            "./topic-model-data/master_papers.csv",
            "./topic-model-data/detected-phrases",
            "--input-format",
            "csv",
            "--text-key",
            "abstract",
            "--id-key",
            "paperId",
            "--lowercase",
            "--min-count",
            "15",
            "--token-regex",
            "wordlike",
            "--no-detect-entities",
        ]
    )


"""Summary: Creates processed folder and calls preprocess from soup-nuts
"""


def run_preprocess():
    subprocess.run(
        [
            "soup-nuts",
            "preprocess",
            "./topic-model-data/master_papers.csv",
            "./topic-model-data/processed",
            "--text-key",
            "abstract",
            "--id-key",
            "paperId",
            "--lowercase",
            "--input-format",
            "csv",
            "--detect-entities",
            "--phrases",
            "./topic-model-data/detected-phrases/phrases.txt",
            "--max-doc-freq",
            "0.9",
            "--min-doc-freq",
            "2",
            "--output-text",
            "--metadata-keys",
            "abstract,title,url",
            "--stopwords",
            "./topic-model-data/stopwords.txt",
        ]
    )


"""Summary: Runs topic model, currently runs all options of topic numbers but feel free to exclude any by commenting them out
"""


def run():
    change_dir_data()

    get_csv()

    run_detect_phrases()

    run_preprocess()

    change_dir_src()

    # 10 topics
    subprocess.run(
        ["python", "run_tomotopy.py", "--num_topics", "10", "--iterations", "1000"]
    )

    # 25 topics
    subprocess.run(
        ["python", "run_tomotopy.py", "--num_topics", "25", "--iterations", "1000"]
    )

    # 50 topics
    subprocess.run(
        ["python", "run_tomotopy.py", "--num_topics", "50", "--iterations", "1000"]
    )