import matplotlib.pyplot as plt
import pandas as pd

file_path = '../data/arxiv_papers.csv'

# Read the CSV file into a DataFrame
arxiv_papers_df = pd.read_csv(file_path)

first_author_counts = arxiv_papers_df['firstAuthor'].value_counts()

sorted_first_author_counts = first_author_counts.sort_values(ascending=False)
top_20_first_authors = sorted_first_author_counts.head(20)

plt.figure(figsize=(12, 8))
top_20_first_authors.plot(kind='bar', rot=30)
plt.title('Publication Frequency by Author')
plt.xlabel('Author')
plt.ylabel('Number of Papers Published')
plt.show()
