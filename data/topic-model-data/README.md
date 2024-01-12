# Running a topic model on the data (MANUAL)

## Installation
First, [install poetry](https://python-poetry.org/docs), then install this package with `poetry install`. It is also possible to just install directly with `pip install -e .`.

Type `soup-nuts --help` to make sure the preprocessing package was installed correctly. If it wasn't, clone [this repo](https://github.com/ahoho/topics) and try running `poetry install` there, then `poetry add tomotopy`.

## Process data

Download the CSV of papers and abstracts.

```console
 curl https://huggingface.co/datasets/PromptSystematicReview/Prompt_Systematic_Review_Dataset/blob/main/master_papers.csv -o master_papers.csv
 ```

Optionally learn common phrases (e.g., `prompt_engineering`):

```bash
mkdir ./detected-phrases

soup-nuts detect-phrases \
    master_papers.csv \
    ./detected-phrases \
    --input-format csv \
    --text-key abstract \
    --id-key paperId \
    --lowercase \
    --min-count 15 \
    --token-regex wordlike \
    --no-detect-entities                                                            
```

Preprocess the data---feel free to play around with these parameters (see `soup-nuts preprocess --help` for information)

```bash
soup-nuts preprocess \
    master_papers.csv\
    ./processed\
    --text-key abstract \
    --id-key paperId \
    --lowercase \
    --input-format csv \
    --detect-entities \
    --phrases ./detected-phrases/phrases.json \
    --max-doc-freq 0.9 \
    --min-doc-freq 2 \
    --output-text \
    --metadata-keys abstract,title,url \
    --stopwords stopwords.txt
```

## Run the topic model

```
python run_tomotopy.py --num_topics 25 --iterations 1000
```

You can view the outputs in `topic_outputs-<num_topics>.html`