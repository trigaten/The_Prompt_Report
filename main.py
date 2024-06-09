from prompt_systematic_review import collect_papers
from prompt_systematic_review import config_data

config_data.DataFolderPath = "./data"
config_data.DotenvPath = "./.env"

### IF RUNNING EXPERIMENTS MULTIPLE TIMES, PLEASE SET 
### hasDownloadedPapers TO True IN config_data.py or uncomment the following line
### config_data.hasDownloadedPapers = True

if not config_data.hasDownloadedPapers:
    collect_papers.collect()
    config_data.hasDownloadedPapers = True


from prompt_systematic_review import experiments
import os

os.makedirs(config_data.DataFolderPath + os.sep + "experiments_output", exist_ok=True)
print("Running experiments...")
for experiment in experiments.experiments:
    try:
        experiment.run()
    except Exception as e:
        print(f"Error running experiment {experiment.__name__}: {e}")
        continue

print("Experiments completed. See data/experiments_output for output files")
