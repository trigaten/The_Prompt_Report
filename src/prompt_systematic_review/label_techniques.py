import pandas as pd
from openai import OpenAI
import json
import os

"""
Script to extract datasets, benchmarks, frameworks, and models from papers
"""
def label_techniques(title, abstract, model="gpt-4"):
  system_message = """
  You are a knowledgeable Natural Language Processing researcher who will
  help me extract datasets, benchmarks, frameworks, and models from papers.
  The user will give you a generative AI research paper title and abstract, and using this information you should
  write to a JSON the names of any mentioned datasets, benchmarks, frameworks, or models.
  The JSON columns should have the following fields: 'datasets', 'benchmarks', 'frameworks', 'models'.

  Examples:

  Input:
  Title: "ReAct: Synergizing Reasoning and Acting in Language Models", Abstract: "While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines, as well as improved human interpretability and trustworthiness over methods without reasoning or acting components. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. On two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples."

  Output in JSON format, NOT string format:
  {datasets: "HotpotQA, Fever", benchmarks: "ALFWorld, WebShop", frameworks: "ReAct", models: ""}
  """

  user_message = f"Title: '{title}', Abstract: '{abstract}'."
  
  client = OpenAI(
    organization=os.environ.get("OPENAI_API_KEY_ORG"),
  )
  
  response = client.chat.completions.create(
    model="gpt-4",
    messages=[
      {"role": "system", "content": system_message},
      {"role": "user", "content": user_message},
    ]
  )

  try:
      content = json.loads(response.choices[0].message.content)
      datasets = content.get("datasets", "N/A")
      benchmarks = content.get("benchmarks", "N/A")
      frameworks = content.get("frameworks", "N/A")
      models = content.get("models", "N/A")
      return {
          "Title": title,
          "Datasets": datasets,
          "Benchmarks": benchmarks,
          "Frameworks": frameworks,
          "Models": models,
      }
  except json.JSONDecodeError as e:
      print(e)
      return {
          "Title": title,
          "Error": "Invalid JSON response",
          "Response": response.choices[0].message.content,
      }


