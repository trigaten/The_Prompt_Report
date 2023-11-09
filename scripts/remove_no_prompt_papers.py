"""Remove papers from arxiv_papers.csv and from arxivPDFs that don't contain the word "prompt""""

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import os
import pandas as pd
from tqdm import tqdm
from PyPDF2 import PdfReader

def filter_and_save_pdfs(folder_path, csv_path, output_csv_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Add a new column "pdf_titles" by extracting titles from the URLs
    df['pdf_titles'] = df['url'].apply(lambda x: os.path.basename(x))

    kept_pdfs = []

    # Iterate through all files in the folder
    for filename in tqdm(os.listdir(folder_path)):
        try:
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)

                # Open the PDF file
                with open(file_path, "rb") as pdf_file:
                    pdf_reader = PdfReader(pdf_file)

                    # Iterate through all pages in the PDF
                    for page_num in range(len(pdf_reader.pages)):
                        # Extract text from the current page
                        text = pdf_reader.pages[page_num].extract_text()

                        # Check if the word "prompt" is present in the text
                        if "prompt" in text.lower():
                            kept_pdfs.append(filename)
                            break  # No need to check other pages if "prompt" is found in one
                    else:
                        # If the loop completes without breaking, the word "prompt" was not found
                        pass
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Create a new DataFrame with entries for the kept PDFs
    kept_df = df[df['pdf_titles'].isin(kept_pdfs)]

    # Save the new DataFrame as a new CSV
    kept_df.to_csv(output_csv_path, index=False)

    return len(kept_pdfs)

# Provide the path to the folder containing PDFs, the original CSV file, and the output CSV file
folder_path = "arxivPDFs"
csv_path = "arxiv_papers.csv"
output_csv_path = "arxiv_papers.csv"

result = filter_and_save_pdfs(folder_path, csv_path, output_csv_path)

print(f"Number of PDFs containing the word 'prompt' and saved to new CSV: {result}")
