from prompt_systematic_review.benchmarking import (
    query_model,
    extract_numbers,
    evaluate_prompts,
    evaluate_mmlu_response,
    response_to_dict,
    load_mmlu,
    find_quotes_with_letters,
    find_parentheses_with_letters,
)

import pytest
from dotenv import load_dotenv
import os
import openai
import pandas as pd
import json


@pytest.fixture(scope="module")
def api_key():
    load_dotenv(dotenv_path="./.env")  # Adjust the path as needed
    return os.getenv("OPENAI_API_KEY")


@pytest.mark.API_test
def test_json_output(api_key):
    openai.api_key = api_key
    prompts = [
        "You are a brilliant math professor. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
        "You are a foolish high-school student. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
    ]

    model = "gpt-3.5-turbo-1106"

    response = query_model(
        prompts[0],
        "What is 4 + 4?",
        model_name=model,
        output_tokens=150,
    )

    response_dict = response_to_dict(response)

    # Check the main keys
    assert "id" in response_dict
    assert "model" in response_dict
    assert "object" in response_dict
    assert "created" in response_dict
    assert "system_fingerprint" in response_dict
    assert "choices" in response_dict
    assert "usage" in response_dict

    # Check the types of the main keys
    assert isinstance(response_dict["id"], str)
    assert isinstance(response_dict["model"], str)
    assert isinstance(response_dict["object"], str)
    assert isinstance(response_dict["created"], int)
    assert isinstance(response_dict["system_fingerprint"], str)
    assert isinstance(response_dict["choices"], list)
    assert isinstance(response_dict["usage"], dict)

    # Check the structure and types of the 'choices' key
    assert len(response_dict["choices"]) > 0
    for choice in response_dict["choices"]:
        assert "finish_reason" in choice
        assert "index" in choice
        assert "message" in choice
        assert isinstance(choice["finish_reason"], str)
        assert isinstance(choice["index"], int)
        assert isinstance(choice["message"], dict)
        assert "content" in choice["message"]
        assert "role" in choice["message"]
        assert isinstance(choice["message"]["content"], str)
        assert isinstance(choice["message"]["role"], str)

    # Check the structure and types of the 'usage' key
    assert "completion_tokens" in response_dict["usage"]
    assert "prompt_tokens" in response_dict["usage"]
    assert "total_tokens" in response_dict["usage"]
    assert isinstance(response_dict["usage"]["completion_tokens"], int)
    assert isinstance(response_dict["usage"]["prompt_tokens"], int)
    assert isinstance(response_dict["usage"]["total_tokens"], int)


@pytest.mark.API_test
def test_query_model(api_key):
    openai.api_key = api_key
    prompt = "You are a brilliant math professor. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible."
    question = "What is 4 + 4?"
    model_name = "gpt-3.5-turbo-1106"
    output_tokens = 150
    response = query_model(prompt, question, model_name, output_tokens)
    assert isinstance(response.choices[0].message.content, str)
    assert len(response.choices[0].message.content) > 0
    assert "8" in response.choices[0].message.content

    prompt = 'You are a brilliant math professor. Solve the following problem and return a JSON with the first entry being the reasoning behind the choice labeled as "reasoning", and the second entry being the answer to the question containing only the letter "A", "B", "C" or "D", labeled as "answer". Try to keep your reasoning concise.'
    question = "What is 4 + 4? A. 8 B. 9 C. 10 D. 11"
    model_name = "gpt-3.5-turbo-1106"
    output_tokens = 150
    json_mode = True
    response = query_model(
        prompt, question, model_name, output_tokens, return_json=json_mode
    )
    json_response = json.loads(response.choices[0].message.content)
    assert isinstance(json_response, dict)
    assert json_response["answer"] == "A"


def test_with_commas_and_dollar_sign():
    assert extract_numbers("####$1,000") == [1000]
    assert extract_numbers("####$1,000.00") == [1000]


def test_with_commas_without_dollar_sign():
    assert extract_numbers("####1,000") == [1000]
    assert extract_numbers("####1,000.00") == [1000]


def test_without_commas_with_dollar_sign():
    assert extract_numbers("####$1000") == [1000]
    assert extract_numbers("####$1000.00") == [1000]


def test_without_commas_and_dollar_sign():
    assert extract_numbers("####1000") == [1000]
    assert extract_numbers("####1000.00") == [1000]


def test_with_spaces():
    assert extract_numbers("#### 1,000") == [1000]
    assert extract_numbers("#### 1000") == [1000]
    assert extract_numbers("#### $1,000") == [1000]
    assert extract_numbers("#### $1000") == [1000]


def test_empty_string():
    assert extract_numbers("") == []


def test_string_without_numbers():
    assert extract_numbers("####NoNumbersHere") == []


def test_multiple_numbers():
    assert extract_numbers("####$1,000 ####$2,000") == [1000, 2000]
    assert extract_numbers("####1000 ####2000") == [1000, 2000]


def test_load_mmlu():
    with open("data/mmlu_configs.json", "r") as file:
        mmlu_configs = json.load(file)["configs"]
    df = load_mmlu(mmlu_configs, "test")

    assert df.iloc[0]["input"] == "When was the telescope invented by Galileo?"
    assert df.iloc[0].A == "1409"
    assert df.iloc[0].B == "1509"
    assert df.iloc[0].C == "1609"
    assert df.iloc[0].D == "1709"
    assert df.iloc[0].answer == "C"
    assert df.iloc[0].config == "astronomy"
    assert len(df) == 13911


def test_modular_prompts():
    with open("data/prompts.json", "r") as file:
        prompts = json.load(file)

    for prompt_name, prompt in prompts.items():
        assert isinstance(prompt, str) or isinstance(prompt, dict)
        assert len(prompt) > 0
        assert isinstance(prompt_name, str)


def test_find_quotes_with_letters():
    answer = "A"
    assert find_quotes_with_letters(answer) == []

    answer = "A. 8"
    assert find_quotes_with_letters(answer) == []

    answer = "'a'"
    assert find_quotes_with_letters(answer) == []

    answer = '"A"'
    assert find_quotes_with_letters(answer) == ["A"]

    answer = "'A''B''C''D'"
    assert find_quotes_with_letters(answer) == ["A", "B", "C", "D"]

    answer = '"A""B""A"'
    assert find_quotes_with_letters(answer) == ["A", "B", "A"]


def test_find_parentheses_between_letters():
    answer = "A"
    assert find_parentheses_with_letters(answer) == []

    answer = "A. 8"
    assert find_parentheses_with_letters(answer) == []

    answer = "'a'"
    assert find_parentheses_with_letters(answer) == []

    answer = '"A"'
    assert find_parentheses_with_letters(answer) == []

    answer = "'A''B''C''D'"
    assert find_parentheses_with_letters(answer) == []

    answer = '"A""B""A"'
    assert find_parentheses_with_letters(answer) == []

    answer = "(A)"
    assert find_parentheses_with_letters(answer) == ["A"]

    answer = "(A)(B)(C)(D)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "C", "D"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]

    answer = "(A)(B)(A)"
    assert find_parentheses_with_letters(answer) == ["A", "B", "A"]


def test_evaluate_mmlu_response():
    response = Response("(A)")
    correct = "A"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"

    response = Response("(A)")
    correct = "B"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "incorrect"

    response = Response("(A)(B)")
    correct = "C"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "incorrect"


def test_evaluate_correct_answer_is():
    response = Response("The correct answer is (B)")
    correct = "B"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"

    response = Response("(A), (C) but the correct answer is (B)")
    correct = "B"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"

    response = Response("(A), (C) but the correct answer is (C)")
    correct = "B"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "incorrect"

    response = Response("(A), (B) but the correct answer is (C)")
    correct = "B"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "incorrect"

    response = Response("(C)")
    correct = "B"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "incorrect"

    response = Response("The answer to The problem is (C)")
    correct = "C"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"

    response = Response("The answer is (C)")
    correct = "C"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"

    response = Response("(C) is the correct option.")
    correct = "C"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"

    response = Response("(C) is the correct answer.")
    correct = "C"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"
    
    response = Response("The telescope was invented by Galileo. To determine when it was invented, we need to consider the options provided.\n\nOption (A): 1409 - This is unlikely as Galileo was born in 1564, so he could not have invented the telescope before he was born.\n\nOption (B): 1509 - This is also unlikely as Galileo was still a child at this time and did not invent the telescope until later in his life.\n\nOption (C): 1609 - This is the most likely option as Galileo invented the telescope in 1609.\n\nOption (D): 1709 - This is too late as Galileo passed away in 1642, so he could not have invented the telescope in 1709.\n\nTherefore, the correct answer is (C): 1609.")
    correct = "C"
    json_mode = False
    assert evaluate_mmlu_response(response, correct, json_mode) == "correct"


class Response:
    def __init__(self, content):
        self.message = Message(content)


class Message:
    def __init__(self, content):
        self.content = content
