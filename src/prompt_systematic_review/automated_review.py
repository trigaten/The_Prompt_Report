import pandas as pd
from openai import OpenAI
import json
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, parse_qs
import os
import tqdm


def review_abstract_title_categorical(
    title: str, abstract: str, model, client: OpenAI.Client, url: str
) -> dict:
    """
    Rate the relevance of a paper to the topic of prompt engineering based on its title and abstract.

    :param title: The title of the paper.
    :type title: str
    :param abstract: The abstract of the paper.
    :type abstract: str
    :param model: The name of the model to use for rating.
    :type model: str
    :param client: The OpenAI client object.
    :type client: OpenAI.Client
    :param url: The URL of the paper.
    :type url: str
    :return: A dictionary containing the title, model, probability, reasoning, and URL.
    :rtype: dict
    """
    system_message = """You are a lab assistant, helping with a systematic review on prompt engineering. You've been asked to rate the relevance of a paper to the topic of prompt engineering.
To be clear, this review will strictly cover hard prefix prompts. For clarification: Hard prompts have tokens that correspond directly to words in the vocab. For example, you could make up a new token by adding two together. This would no longer correspond to any word in the vocabulary, and would be a soft prompt
Prefix prompts are prompts used for most modern transformers, where the model predicts the words after this prompt. In earlier models, such as BERT, models could predict words (e.g. <MASK>) in the middle of the prompt. Your job is to be able to tell whether a paper is related to (or simply contains) hard prefix prompting or prompt engineering. Please note that a paper might not spell out that it is using "hard prefix" prompting and so it might just say prompting. In this case, you should still rate it as relevant to the topic of prompt engineering. 
Please also note, that a paper that focuses on training a model as opposed to post-training prompting techniques is considered irrelevant. Provide a response in JSON format with two fields: 'rating' (a string that is one of the following categories: 'highly relevant', 'somewhat relevant', 'neutrally relevant', 'somewhat irrelevant', 'highly irrelevant') indicating relevance to the topic of prompt engineering) and 'reasoning' (that justifies your reasoning)"""

    user_message = f"Title: '{title}', Abstract: '{abstract}'. Rate its relevance to the topic of prompt engineering as one of the following categories: 'highly relevant', 'somewhat relevant', 'neutrally relevant', 'somewhat irrelevant', 'highly irrelevant',  and provide text from the abstract that justifies your reasoning"

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    try:
        content = json.loads(response.choices[0].message.content)
        probability = content.get("rating", "Not provided")
        reasoning = content.get("reasoning", "No reasoning provided")
        return {
            "Title": title,
            "Model": model,
            "Probability": probability,
            "Reasoning": reasoning,
            "Url": url,
        }
    except json.JSONDecodeError:
        return {
            "Title": title,
            "Model": model,
            "Error": "Invalid JSON response",
            "Response": response.choices[0].message.content,
        }
