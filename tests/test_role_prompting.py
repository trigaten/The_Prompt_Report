from src.prompt_systematic_review.role_prompting import (
    query_model,
    extract_numbers,
    evaluate_prompts,
    evaluate_response,
)
import pytest
from dotenv import load_dotenv
import os
import openai


@pytest.fixture(scope="module")
def api_key():
    load_dotenv(dotenv_path="./.env")  # Adjust the path as needed
    return os.getenv("OPENAI_API_KEY")


@pytest.mark.API_test
def test_query_model(api_key):
    openai.api_key = api_key
    prompt = "You are a brilliant math professor. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible."
    question = "What is 4 + 4?"
    model_name = "gpt-4"
    output_tokens = 150
    response = query_model(prompt, question, model_name, output_tokens)
    assert isinstance(response, str)
    assert len(response) > 0
    assert "8" in response


@pytest.mark.API_test
def test_query_model(api_key):
    openai.api_key = api_key
    prompts = [
        "You are a brilliant math professor. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
        "You are a foolish high-school student. Solve the following problem and put your answer after four hashtags like the following example: \nQuestion: What is 4 + 4?\nAnswer: 4 + 4 is ####8\n\n Make your response as short as possible.",
    ]

    dataset = "gsm8k"
    config_name = "main"
    split = "test"
    model = "gpt-4"
    examples = 1

    eval = evaluate_prompts(
        prompts,
        dataset,
        config_name,
        split,
        model,
        examples,
    )
    assert isinstance(eval, dict)
    assert len(eval) == 2


def test_evaluate_response():
    class Response:
        def __init__(self, content):
            self.message = Message(content)

    class Message:
        def __init__(self, content):
            self.content = content

    response = Response("####8")
    correct_answer = "####8"
    assert evaluate_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####9"
    assert evaluate_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_response(response, correct_answer) == True

    response = Response("####8")
    correct_answer = "####8.1"
    assert evaluate_response(response, correct_answer) == False

    response = Response("####8")
    correct_answer = "####8.0"
    assert evaluate_response(response, correct_answer) == True


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
