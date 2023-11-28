import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

file_path = "./master_papers.csv"

arxiv_papers_df = pd.read_csv(file_path)

# Convert to datetime
arxiv_papers_df["dateSubmitted"] = pd.to_datetime(
    arxiv_papers_df["dateSubmitted"], format="%Y-%m-%d", errors="coerce"
)


earliest_date = arxiv_papers_df["dateSubmitted"].min()
latest_date = arxiv_papers_df["dateSubmitted"].max()

# count occurrences of each submission date
submission_dates_counts = arxiv_papers_df["dateSubmitted"].value_counts().sort_index()

# Plot 1: number of papers submitted over time, by year
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(
    submission_dates_counts.index,
    submission_dates_counts.values,
    marker="o",
    linestyle="-",
    color="blue",
)
ax.set_title("Number of Papers Submitted Over Time")
ax.set_xlabel("Submission Date")
ax.set_ylabel("Number of Papers Submitted")
ax.grid(True)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))

ax.set_xlim(earliest_date, latest_date)
plt.xticks(rotation=45, ha="right")

plt.show()


# Plot 2: number of papers submitted over time, between 2021 and 2023 (majority of papers)
# Filter papers between 2021 and 2023
filtered_df = arxiv_papers_df[
    (arxiv_papers_df["dateSubmitted"].dt.year >= 2021)
    & (arxiv_papers_df["dateSubmitted"].dt.year <= 2023)
]

earliest_date = filtered_df["dateSubmitted"].min()
latest_date = filtered_df["dateSubmitted"].max()

submission_dates_counts = filtered_df["dateSubmitted"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(
    submission_dates_counts.index,
    submission_dates_counts.values,
    marker="o",
    linestyle="-",
    color="blue",
)
ax.set_title("Number of Papers Submitted Between 2021 and 2023")
ax.set_xlabel("Submission Date")
ax.set_ylabel("Number of Papers Submitted")
ax.grid(True)

ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%Y"))

ax.set_xlim(earliest_date, latest_date)
plt.xticks(rotation=45, ha="right")
plt.show()


# Plot 3: number of papers submitted over time, between 2021 and 2023,
# with vertical lines indicating release dates of different LLMs

chatgpt_release_date = pd.to_datetime("2022-11-30")
copilot_release_date = pd.to_datetime("2023-02-07")
llama_release_date = pd.to_datetime("2023-02-24")
bard_release_date = pd.to_datetime("2023-03-21")
dolly_release_date = pd.to_datetime("2023-04-12")

fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(
    submission_dates_counts.index,
    submission_dates_counts.values,
    marker="o",
    linestyle="-",
    color="blue",
)
ax.set_title("Number of Papers Submitted Between 2021 and 2023")
ax.set_xlabel("Submission Date")
ax.set_ylabel("Number of Papers Submitted")
ax.grid(True)

ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d-%Y"))

ax.set_xlim(earliest_date, latest_date)

plt.xticks(rotation=45, ha="right")

# Add vertical lines to indicate release dates of different LLMs
ax.axvline(x=chatgpt_release_date, color="red", linestyle="--", label="ChatGPT Release")
ax.axvline(x=dolly_release_date, color="green", linestyle="--", label="Dolly Release")
ax.axvline(
    x=llama_release_date, color="darkorange", linestyle="--", label="LLaMA Release"
)
ax.axvline(
    x=copilot_release_date,
    color="purple",
    linestyle="--",
    label="Microsoft Copilot Release",
)
ax.axvline(
    x=bard_release_date, color="brown", linestyle="--", label="Google Bard Release"
)

plt.legend()

plt.show()
