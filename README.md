# Prompt Engineering Survey

## Install requirements

after cloning, run `pip install -r requirements.txt` from root

## Set up API keys

Make a file at root called `.env`.

For HF: https://huggingface.co/docs/hub/security-tokens, also run `huggingface-cli login`

Put your key in like:

`OPENAI_API_KEY=sk-...`
`SEMANTIC_SCHOLAR_API_KEY=...`
`HF_TOKEN=...`

Then to load the .env file, type:
pip install pytest-dotenv

You can also choose to update the env file by doing:
py.test --envfile path/to/.env

In the case that you have several .env files, create a new env_files in the pytest config folder and type:

env_files =
.env
.test.env
.deploy.env

## blacklist.csv

Papers we should not include due to being poorly written or AI generated

## Notes

- Sometimes a paper title may appear differently on the arXiv API. For example, "Visual Attention-Prompted Prediction and Learning" (arXiv:2310.08420), according to arXiv API is titled "A visual encoding model based on deep neural networks and transfer learning"

- When testing APIs, there may be latency and aborted connections

- Publication dates of papers from IEEE are missing the day about half the time. They also may come in any of the following formats
  - "April 1988"
  - "2-4 April 2002"
  - "29 Nov.-2 Dec. 2022"
