from prompt_systematic_review import collect_papers
from prompt_systematic_review import config_data

config_data.DataFolderPath = "./data"
config_data.DotenvPath = "./.env"
if not config_data.hasDownloadedPapers:
    collect_papers.collect()
    config_data.hasDownloadedPapers = True


# from prompt_systematic_review import experiments
# import os

# os.makedirs(config_data.DataFolderPath + os.sep + "experiments_output", exist_ok=True)

# for experiment in experiments.experiments:
#     experiment.run()
