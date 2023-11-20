import pandas as pd


def clean_duplicates(input, output):
    """
    Description:
    This function is used for removing duplicate entries from a dataset based on the 'Title' and 'First Author' columns.
    It handles two scenarios: processing bulk data and processing regular data.
    Parameters:
    bulk (bool): A flag to determine if the function should process bulk data (default is False).
    """
    df = pd.read_csv(input)
    df_cleaned = df.drop_duplicates(subset=["Title", "Authors"])
    df_cleaned.to_csv(output, index=False)


def clean_against_previous_dataset(current, previous, new):
    """
    Description:
      This function cleans a current dataset by removing entries that are already present in a previous dataset.
      It specifically focuses on the 'Title' field to identify duplicates.
    Parameters:
      current (str): Filename of the current dataset.
      previous (str): Filename of the previous dataset to compare against.
      new (str): Filename for saving the cleaned dataset.
    """
    current_df = pd.read_csv(current)
    previous_df = pd.read_csv(previous)

    current_df["Title"] = current_df["Title"].str.lower()
    previous_df["title"] = previous_df["title"].str.lower()

    unique_titles = ~current_df["Title"].isin(previous_df["title"])

    cleaned_df = current_df[unique_titles]

    cleaned_df.to_csv(new, index=False)
