import os
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from prompt_systematic_review.config_data import DataFolderPath

def keyword_wordcloud():
    """
    Generate a word cloud from the abstracts of the papers in the master_papers.csv file.

    This function reads the abstracts from the master_papers.csv file, concatenates them into a single
    string, and generates a word cloud visualization using the WordCloud library. The resulting word
    cloud is saved as an image file.

    :return: None
    :rtype: None
    """
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
            DataFolderPath,
            "experiments_output" + os.sep + "keyword_wordcloud_output.png",
        )
    )


class Experiment:
    def run():
        keyword_wordcloud()

if __name__ == "__main__":
    keyword_wordcloud()
