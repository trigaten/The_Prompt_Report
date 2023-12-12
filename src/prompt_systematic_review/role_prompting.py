from prompt_systematic_review.load_hf_data import load_hf_dataset
import openai
from typing import List
import re
import time
import json
from json.decoder import JSONDecodeError
from datetime import datetime
from tenacity import (
    retry,
    before_log,
    stop_after_attempt,
    wait_random_exponential,
)
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("data/mmlu_configs.json", "r") as file:
    mmlu_configs = json.load(file)["configs"]


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(20),
    before=before_log(logger, logging.INFO),
)
def query_model_with_backoff(**kwargs):
    try:
        return openai.chat.completions.create(**kwargs)
    except Exception as e:
        logger.error(f"Query failed with error: {e}")
        raise


def query_model(
    prompt: str,
    question: str,
    model_name: str,
    output_tokens: int = 500,
    return_json=False,
    rereading: bool = False,
    seed: int = 42,
    temperature: float = 0.0,
) -> dict:
    """
    Query the OpenAI API with a timeout.
    :param prompt: The prompt to use.
    :param question: The question to use from the dataset.
    :param model_name: The OpenAI model to use.
    :param output_tokens: The maximum number of output tokens to generate.
    :param timeout: Timeout for the request in seconds.
    :return: The response from the API or None if timeout occurs.
    """
    if rereading:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question + "\n\n" + question},
        ]
    else:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]
    if return_json:
        response = query_model_with_backoff(
            model=model_name,
            messages=messages,
            max_tokens=output_tokens,
            response_format={"type": "json_object"},
            seed=seed,
            temperature=temperature,
        )
        return response
    else:
        response = query_model_with_backoff(
            model=model_name,
            messages=messages,
            max_tokens=output_tokens,
            seed=seed,
            temperature=temperature,
        )
        return response


def evaluate_gsm8k_response(response: dict, correct_answer: str) -> bool:
    """
    Evaluate the response from the API for a GSM8K question and return whether it is correct.
    :param response: The response from the API.
    :param correct_answer: The correct answer to the question taken from the dataset.
    :return: Whether the response is correct.
    """
    marked_nums_in_response = extract_numbers(response.message.content)
    if len(marked_nums_in_response) == 0:
        return False
    else:
        final_answer = extract_numbers(response.message.content)[-1]
    correct = extract_numbers(correct_answer)[-1]
    return final_answer == correct


def evaluate_mmlu_response(response: dict, correct_answer: str) -> bool:
    """
    Evaluate the response from the API for a MMLU question and return whether it is correct.
    :param response: The response from the API.
    :param correct_answer: The correct answer to the question taken from the dataset.
    :return: Whether the response is correct.
    """
    try:
        json_response = json.loads(response.message.content)
        return json_response["answer"] == correct_answer
    except JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        print("Error occurred at: Line {}, Column {}".format(e.lineno, e.colno))
        print(
            "Problematic text snippet: ",
            response.message.content[max(0, e.pos - 50) : e.pos + 50],
        )
        return False
    # json_response = json.loads(response.message.content)
    # return json_response["answer"] == correct_answer


def evaluate_prompts(
    prompts: List[str],
    dataset: str,
    config_name: str,
    split: str,
    model_name: str,
    examples: None or int = 1,
    start_index: int = 0,
    log_interval: int = 25,
    max_tokens: int = 5000,
    json_mode: bool = False,
    reread: bool = False,
    seed: int = 42,
    temperature: float = 0.0,
) -> dict:
    """
    Evaluate a list of prompts on a dataset and return the results.
    :param prompts: The prompts to use.
    :param dataset: The dataset to use. This will be "gsm8k" for the GSM-8k dataset.
    :param config_name: The configuration name to use. This will be "main" for the GSM-8k dataset.
    :param split: The split of the dataset to use. One of the splits for the GSM-8k dataset is "test".
    :param model_name: The OpenAI model to use (ex. "gpt-4").
    :param examples: The number of examples to evaluate, 1 by default.
    :return: The results of the evaluation.
    """

    query_count = 0
    results = {prompt: {"correct": 0, "total": 0} for prompt in prompts}
    information = {
        "dataset": dataset,
        "config_name": config_name,
        "split": split,
        "model_name": model_name,
        "examples": examples,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "calls": [],
        "total_wall_time": 0,
    }

    if dataset == "gsm8k":
        data = load_hf_dataset(dataset_name=dataset, name=config_name, split=split)

        for i, item in enumerate(data):
            if i >= start_index:
                question = item["question"]
                correct_answer = item["answer"]
                for prompt in prompts:
                    start_time = time.time()
                    response = query_model(
                        prompt,
                        question,
                        model_name=model_name,
                        output_tokens=max_tokens,
                        return_json=json_mode,
                        rereading=reread,
                        seed=seed,
                        temperature=temperature,
                    )
                    query_count += 1
                    end_time = time.time()
                    wall_time = end_time - start_time
                    information["total_wall_time"] += wall_time
                    information["total_input_tokens"] += response.usage.prompt_tokens
                    information[
                        "total_output_tokens"
                    ] += response.usage.completion_tokens
                    response_dict = response_to_dict(response)
                    is_correct = evaluate_gsm8k_response(
                        response.choices[0], correct_answer
                    )
                    information["calls"].append(
                        {
                            "prompt": prompt,
                            "question": question,
                            "correct_answer": correct_answer,
                            "response": response_dict,
                            "marked_correct": is_correct,
                            "wall_time": wall_time,
                        }
                    )
                    results[prompt]["total"] += 1
                    if is_correct:
                        results[prompt]["correct"] += 1
                    if query_count % log_interval == 0:
                        try:
                            write_to_file(
                                [results, information], query_count, log_interval
                            )
                        except Exception as e:
                            print(f"Error writing to file: {e}")

            if examples and i + 1 == examples + start_index:
                break
        return results, information

    elif dataset == "mmlu":
        df = load_mmlu(configs=mmlu_configs, split=split)

        for i, example in df.iterrows():
            if i >= start_index:
                question = example["input"]
                correct_answer = example["target"]
                choice_A, choice_B, choice_C, choice_D = (
                    example["A"],
                    example["B"],
                    example["C"],
                    example["D"],
                )
                choices = {
                    "A": choice_A,
                    "B": choice_B,
                    "C": choice_C,
                    "D": choice_D,
                }
                multiple_choice_question = """
                {question}
                A. {choice_A}
                B. {choice_B}
                C. {choice_C}
                D. {choice_D}
                """.format(
                    question=question,
                    choice_A=choice_A,
                    choice_B=choice_B,
                    choice_C=choice_C,
                    choice_D=choice_D,
                )
                for prompt in prompts:
                    start_time = time.time()
                    response = query_model(
                        prompt,
                        multiple_choice_question,
                        model_name=model_name,
                        output_tokens=max_tokens,
                        return_json=json_mode,
                        rereading=reread,
                    )
                    end_time = time.time()
                    wall_time = end_time - start_time
                    query_count += 1
                    information["total_wall_time"] += wall_time
                    information["total_input_tokens"] += response.usage.prompt_tokens
                    information[
                        "total_output_tokens"
                    ] += response.usage.completion_tokens
                    response_dict = response_to_dict(response)
                    is_correct = evaluate_mmlu_response(
                        response.choices[0], correct_answer
                    )

                    information["calls"].append(
                        {
                            "prompt": prompt,
                            "question": "Question: "
                            + multiple_choice_question
                            + "\n Read the question again: "
                            + multiple_choice_question
                            if reread
                            else multiple_choice_question,
                            "correct_answer": correct_answer,
                            "response": response_dict,
                            "marked_correct": is_correct,
                            "wall_time": wall_time,
                            "config": example["config"],
                        }
                    )
                    results[prompt]["total"] += 1
                    if is_correct:
                        results[prompt]["correct"] += 1
                    if query_count % log_interval == 0:
                        try:
                            write_to_file(
                                [results, information], query_count, log_interval
                            )
                        except Exception as e:
                            print(f"Error writing to file: {e}")
            if examples and i + 1 == examples + start_index:
                break
        return results, information

    else:
        # Throw an error saying that we don't support this dataset
        raise NotImplementedError(f"Dataset {dataset} is not supported.")


def extract_numbers(string: str) -> List[int]:
    """
    Extract the number from a string that can take any of the following forms:
    "####1,000", "####1,000.00", "####$1,000", "####$1,000.00", "####1000", "####1000.00", "####$1000", "####$1000.00", "#### 1,000", "#### 1,000.00", "#### 1000", "#### 1000.00"
    param string: The string to extract the number from.
    return: The extracted number.
    """
    # Remove commas from the string
    string_without_commas = string.replace(",", "")

    # Regular expression to find the pattern of three hashtags followed by an optional space or dollar sign, and then a number
    pattern = r"###\s?[\$]?\s?(\d+(?:\.\d+)?)"

    # Find all matches of the pattern in the string without commas
    matches = re.findall(pattern, string_without_commas)

    numbers = [float(match) for match in matches]

    return numbers


def response_to_dict(response):
    # Extract relevant data from the response
    response_data = {
        "id": response.id,
        "model": response.model,
        "object": response.object,
        "created": response.created,
        "system_fingerprint": response.system_fingerprint,
        "choices": [
            {
                "finish_reason": choice.finish_reason,
                "index": choice.index,
                "message": {
                    "content": choice.message.content,
                    "role": choice.message.role,
                },
            }
            for choice in response.choices
        ],
        "usage": {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }

    return response_data


def write_to_file(data, count, log_interval=25):
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = f"data/benchmarking/RP_eval_results_{current_datetime}_part_{((count//log_interval))}.json"
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)
    print(f"Written results to {file_path}")


def load_mmlu(configs: List[str], split: str) -> pd.DataFrame:
    combined_dataset = None
    for config in configs:
        dataset = load_hf_dataset("lukaemon/mmlu", config, split=split)

        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset)

        df["config"] = config

        if combined_dataset is None:
            combined_dataset = df
        else:
            combined_dataset = pd.concat([combined_dataset, df], ignore_index=True)

    df = combined_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    return df
