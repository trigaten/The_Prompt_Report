from load_hf_data import load_hf_dataset
from openai import OpenAI
from typing import List

# What prompts should look like for a dataset like GSM8K:
#     "As an expert mathematician, solve the following and indicate your final answer with '####': ",
#     "Solving step by step, then write your final answer after '####': ",
#     "As a math teacher, explain and conclude your final answer with '####': ",


def query_model(
    key: str, prompt: str, question: str, model_name: str, output_tokens: int = 150
) -> str:
    client = OpenAI(
        api_key=key,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        max_tokens=output_tokens,
    )

    return response.choices[0]


def evaluate_response(response: str, correct_answer: str) -> bool:
    final_answer = response.split("####")[-1].strip()  # Works for the GSM8K dataset
    return final_answer == correct_answer


def evaluate_prompts(
    API_key: str,
    prompts: List[str],
    dataset: str,
    config_name: str,
    split: str,
    model_name: str,
) -> dict:
    dataset = load_hf_dataset(dataset_name=dataset, name=config_name, split=split)

    results = {prompt: {"correct": 0, "total": 0} for prompt in prompts}

    for entry in dataset:
        question = entry["question"]
        correct_answer = entry["answer"]
        for prompt in prompts:
            response = query_model(API_key, prompt, question, model_name=model_name)
            is_correct = evaluate_response(response, correct_answer)
            results[prompt]["total"] += 1
            if is_correct:
                results[prompt]["correct"] += 1

    for prompt, result in results.items():
        accuracy = result["correct"] / result["total"]
        print(f"Prompt: {prompt}\nAccuracy: {accuracy:.2f}\n")
