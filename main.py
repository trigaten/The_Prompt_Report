from prompt_systematic_review import collect_papers
from prompt_systematic_review import config_data


config_data.DataFolderPath = "./data"
config_data.DotenvPath = "./.env"
if not config_data.hasDownloadedPapers:
    collect_papers()


from prompt_systematic_review import experiments

for experiment in experiments.experiments:
    experiment.run()
