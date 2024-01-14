## TopicGPT

### Setup 
- Set your API key in an environment variable called `OPENAI_API_KEY`, or directly in the script/utils.py file. 
- Install the requirements: `pip install -r requirements.txt`

## Usage
- Run as an experiment, file in `src/prompt_systematic_review/experiments`.
- Prompts to generate the topics are in `data/topic-gpt-data/prompt/`.

## Results
- The generated topics are in `data/topic-gpt-data/master_paper_*.md`.
- (Text/Generated topics) pairs are in `data/topic-gpt-data/generation_1_paper.jsonl`.