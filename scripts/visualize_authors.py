import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm

file_path = './master_papers.csv'

# Read the CSV file into a DataFrame
arxiv_papers_df = pd.read_csv(file_path)

# Split authors into a list and explode them to have one row per author
authors = arxiv_papers_df['authors'].str.replace("'", "").str.replace("]", "").str.replace("[", "").str.split(',').explode()

# Count the occurrences of each author
author_counts = authors.value_counts()

# make a vertical bar chart displaying top 20 authors publicaiton counts
sorted_author_counts = author_counts.sort_values(ascending=False)
top_20_authors = sorted_author_counts.head(20)

plt.figure(figsize=(15, 8))
top_20_authors.plot(kind='bar', rot=30, color='blue', edgecolor='black')
plt.title('Publication Frequency by Author (Top 20)')
plt.xlabel('Author')
plt.ylabel('Number of Papers Included in Dataset')
plt.xticks(rotation=45, ha='right')
plt.yticks(range(0, max(top_20_authors) + 2, 2))
plt.tight_layout()
plt.show()

# Display the frequency table
publication_counts = author_counts.values

frequency_table = pd.Series(publication_counts).value_counts().reset_index()

# Rename the columns for clarity
frequency_table.columns = ['Publication Count', 'Frequency']

# Sort the DataFrame by the number
frequency_table = frequency_table.sort_values(by='Publication Count')

# Display the frequency table
print(frequency_table)

# # make a histogram of the number of publications per author
# plt.figure(figsize=(12, 8))
# plt.hist(author_counts, bins=10, alpha=0.7, color='blue', edgecolor='black')

# # Fit a normal distribution to the data
# mu, std = norm.fit(author_counts)

# # Plot the PDF of the fitted distribution
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)

# plt.yscale('log')

# plt.title('Histogram of Number of Publications per Author')
# plt.xlabel('Number of Publications')
# plt.ylabel('Frequency')

# # Add a legend
# plt.legend(['Fit results (mu = %.2f,  std = %.2f)' % (mu, std)])

# # Show the plot
# plt.show()
