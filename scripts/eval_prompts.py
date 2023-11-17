from src.prompt_systematic_review.role_prompting import evaluate_prompts

"""
Test a set of prompts against a dataset and return the results. Currently working for GSM-8k. You must add your openAI API key to the key variable below.
"""

key = ""

prompts = [
    "You are a brilliant math professor. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
    "You are a foolish high-school student. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
]

dataset = "gsm8k"
config_name = "main"
split = "test"
model = "gpt-4"
examples = 10

eval = evaluate_prompts(
    key,
    prompts,
    dataset,
    config_name,
    split,
    model,
    examples,
)
