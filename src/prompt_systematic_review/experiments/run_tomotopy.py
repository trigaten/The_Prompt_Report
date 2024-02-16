# %% imports
import argparse
import html
import subprocess
import os

import tomotopy as tp
import pandas as pd
import numpy as np

from tqdm import tqdm
from prompt_systematic_review.config_data import DataFolderPath


"""
Runs topic model through function run_topic_model()
Running manually is also an option (check README in data/topic-model-data)
"""


# %%
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


"""
    Summary: Show the top docs per topic
"""


# %% create the html content
def retrieve_top_docs_for_topic(theta, topic_idx, min_p=0.1):
    doc_idx_sorted = (-theta[:, topic_idx]).argsort()
    theta_sorted = theta[doc_idx_sorted, topic_idx]
    above_min = theta_sorted > min_p
    if above_min.sum() == 0:
        return doc_idx_sorted[:1], theta_sorted[:1]
    else:
        return doc_idx_sorted[above_min], theta_sorted[above_min]


"""Summary: Downloads master_papers.csv to data folder
"""


def get_csv():
    df = pd.read_csv(
        "https://huggingface.co/datasets/PromptSystematicReview/Prompt_Systematic_Review_Dataset/raw/main/master_papers.csv"
    )
    df.to_csv(
        os.path.join(DataFolderPath, "master_papers.csv"),
        sep=",",
        index=False,
        encoding="utf-8",
    )


"""Summary: Creates detected-phrases folder and calls detect phrases from soup-nuts
"""


def run_detect_phrases():
    os.makedirs(
        os.path.join(DataFolderPath, "topic-model-data/detected-phrases"), exist_ok=True
    )

    subprocess.run(
        [
            "soup-nuts",
            "detect-phrases",
            os.path.join(DataFolderPath, "master_papers.csv"),
            os.path.join(DataFolderPath, "topic-model-data/detected-phrases"),
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
            os.path.join(DataFolderPath, "master_papers.csv"),
            os.path.join(DataFolderPath, "topic-model-data/processed"),
            "--text-key",
            "abstract",
            "--id-key",
            "paperId",
            "--lowercase",
            "--input-format",
            "csv",
            "--detect-entities",
            "--phrases",
            os.path.join(
                DataFolderPath, "topic-model-data/detected-phrases/phrases.txt"
            ),
            "--max-doc-freq",
            "0.9",
            "--min-doc-freq",
            "2",
            "--output-text",
            "--metadata-keys",
            "abstract,title,url",
            "--stopwords",
            os.path.join(DataFolderPath, "topic-model-data/stopwords.txt"),
        ]
    )


"""Summary: Runs the topic model

    Parameters:
        output_path: str - Relative folder path from Prompt_Systematic_Review
        num_topics: str - Number of topics
        iterations: str - # of iterations
        top_words_to_display: str
        top_docs_to_display: str
        parser: argparse.ArgumentParser object - mainly for manual running purpopses
"""


def topic_model(
    output_path="data",
    num_topics="35",
    iterations="1000",
    top_words_to_display="15",
    top_docs_to_display="5",
    parser=None,
):
    running_on_terminal = True
    if parser == None:
        running_on_terminal = False
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input_fname",
            type=str,
            default=os.path.join(
                DataFolderPath, "topic-model-data/processed/train.metadata.jsonl"
            ),
        )
        parser.add_argument("--num_topics", type=int, default=35)
        parser.add_argument("--iterations", type=int, default=1000)
        parser.add_argument("--top_words_to_display", type=int, default=15)
        parser.add_argument("--top_docs_to_display", type=int, default=5)
        parser.add_argument("--min_doc_prob_to_display", type=float, default=None)

    # if in an interactive session, use defaults
    if is_notebook():
        args = parser.parse_args([])
    else:
        if running_on_terminal == False:
            args = parser.parse_args(
                args=[
                    "--num_topics",
                    num_topics,
                    "--iterations",
                    iterations,
                    "--top_words_to_display",
                    top_words_to_display,
                    "--top_docs_to_display",
                    top_docs_to_display,
                ]
            )
        else:
            args = parser.parse_args()

    if args.min_doc_prob_to_display is None:
        args.min_doc_prob_to_display = 1 / args.num_topics

    # %% load data
    data = pd.read_json(args.input_fname, lines=True)

    lda = tp.LDAModel(k=args.num_topics, seed=42)
    for doc in data["tokenized_text"]:
        lda.add_doc(doc.split())

    # %% train model
    with tqdm(range(0, args.iterations, 10)) as pbar:
        for i in range(0, args.iterations, 10):
            lda.train(10)
            pbar.update(1)
            pbar.set_postfix({"log-likelihood": lda.ll_per_word})

    # %% print out a summary
    print(lda.summary())

    # %% create the html template we will populate
    html_template = """  
    <!DOCTYPE html>  
    <html lang="en">  
    <head>  
        <meta charset="UTF-8">  
        <meta name="viewport" content="width=device-width, initial-scale=1.0">  
        <title>Topics and Documents</title>  
        <style>  
            body {{  
                font-family: Arial, sans-serif;  
            }}  
            .topic-container {{  
                margin-bottom: 20px;  
                border-bottom: 1px solid #ccc;  
                padding-bottom: 10px;  
            }}  
            .topic {{  
                display: flex;  
                align-items: center;  
            }}  
            .topic-checkbox {{  
                margin-right: 10px;  
            }}  
            details {{  
                margin-left: 30px;  
            }}  
            summary {{  
                cursor: pointer;  
            }}  
            ul {{  
                list-style-type: none;  
                padding-left: 0;  
            }}  
            li {{  
                background-color: #f0f0f0;  
                border: 1px solid #ccc;  
                border-radius: 4px;  
                padding: 5px;  
                margin-bottom: 5px;  
            }}  
        </style>  
    </head>  
    <body>  
        <div id="content">  
            {content}  
        </div>  
    </body>  
    </html>  
    """

    # %% create the document-topic distribution matrix
    theta = np.array([doc.get_topic_dist() for doc in lda.docs])
    docs = data.to_dict(orient="records")

    # %% create the html content
    content = []
    # for each topic, get the top words and top documents
    for topic_idx in range(args.num_topics):
        # get the top words
        topic = " ".join(
            [
                word
                for word, _ in lda.get_topic_words(
                    topic_idx, top_n=args.top_words_to_display
                )
            ]
        )
        expected_alpha = lda.alpha[topic_idx] / lda.alpha.sum()

        topic_html = f'<div class="topic-container">'
        topic_html += f'<div class="topic"><b>{topic_idx}</b> | {expected_alpha:0.1%}: {topic}</div>'
        topic_html += f"<details><summary>Documents</summary><ul>"

        # get the top documents
        doc_idxs, probs = retrieve_top_docs_for_topic(
            theta, topic_idx, min_p=args.min_doc_prob_to_display
        )
        d_n = max(1, int(round(len(doc_idxs) / args.top_docs_to_display)))

        for idx, p in zip(doc_idxs[::d_n], probs[::d_n]):
            # get the title, abstract, and url; format as html
            doc = docs[idx]
            title, abstract, url = doc["title"], doc["abstract"], doc["url"]
            topic_html += f"<li><a href={url}>{html.escape(title)}</a> ({p:0.1%})</li>"
            # also optionally expand to include the abstract
            topic_html += f"<details><summary>Abstract</summary><p>{html.escape(abstract)}</p></details>"

        topic_html += f"</ul></details></div>"
        content.append(topic_html)

    review_html = html_template.format(content="".join(content))

    # %% write the html to a file
    with open(
        os.path.join(
            DataFolderPath,
            f"experiments_output/topic_outputs-{args.num_topics}.html",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(review_html)

    # %%
    data_by_topic = []
    for topic_idx in range(args.num_topics):
        doc_idxs, probs = retrieve_top_docs_for_topic(
            theta, topic_idx, min_p=1 / args.num_topics
        )
        data_by_topic.append(data.iloc[doc_idxs].assign(topic=topic_idx, prob=probs))

    pd.concat(data_by_topic).to_csv(
        os.path.join(
            DataFolderPath,
            f"experiments_output/topic_outputs-{args.num_topics}.csv",
        ),
        index=False,
    )


"""Summary: Module to be imported to run topic model. Output is routed to data folder by default.

    Parameters:
        output_path: str - Relative folder path from Prompt_Systematic_Review
        num_topics: str - Number of topics
        iterations: str - # of iterations
        top_words_to_display: str
        top_docs_to_display: str
"""


def run_topic_model(
    output_path="data",
    num_topics="35",
    iterations="1000",
    top_words_to_display="15",
    top_docs_to_display="5",
):
    get_csv()
    run_detect_phrases()
    run_preprocess()

    topic_model(
        output_path,
        num_topics,
        iterations,
        top_words_to_display,
        top_docs_to_display,
    )


class Experiment:
    def run():
        run_topic_model()


# If running manually, check README in data/topic-model-data
if __name__ == "__main__":
    # %% arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_fname",
        type=str,
        default=os.path.join(
            DataFolderPath, "topic-model-data/processed/train.metadata.jsonl"
        ),
    )
    parser.add_argument("--num_topics", type=int, default=35)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--top_words_to_display", type=int, default=15)
    parser.add_argument("--top_docs_to_display", type=int, default=5)
    parser.add_argument("--min_doc_prob_to_display", type=float, default=None)

    topic_model(parser=parser)
