"""
This script generates a wordcloud of the most common 
words in abstracts in the master_papers.csv dataset.
"""

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import os
from prompt_systematic_review.config_data import DataFolderPath


def keyword_wordcloud():
    file_path = os.path.join(DataFolderPath, "master_papers.csv")

    # Read the CSV file into a DataFrame
    arxiv_papers_df = pd.read_csv(file_path)

    abstracts = arxiv_papers_df["abstract"]

    full_text = ""
    for index, abstract in abstracts.items():
        full_text += str(abstract) + " "

    wordcloud = WordCloud(
        background_color="white",
        max_words=200,
        width=800,
        height=400,
        colormap="viridis",
        max_font_size=80,
        contour_width=1,
        contour_color="black",
    ).generate(full_text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(
        os.path.join(
            DataFolderPath, "experiments_output" + os.sep + "keyword_wordcloud_output.png"
        )
    )


class Experiment:
    def run():
        keyword_wordcloud()
