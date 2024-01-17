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
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(
    logging.WARNING
)  # Ensure success messages from httpx are not printed to console

with open("data/mmlu_configs.json", "r") as file:
    mmlu_configs = json.load(file)["configs"]


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(20),
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
            {"role": "user", "content": prompt + question + "\n\n" + question},
        ]
    else:
        messages = [
            {"role": "user", "content": prompt + question},
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


def evaluate_mmlu_response(
    response: dict, correct_answer: str, json_mode: bool
) -> bool:
    """
    Evaluate the response from the API for a MMLU question and return whether it is correct.
    :param response: The response from the API.
    :param correct_answer: The correct answer to the question taken from the dataset.
    :return: Whether the response is correct.
    """
    if json_mode:
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
    else:
        all_letters_in_response = find_parentheses_with_letters(
            response.message.content
        )
        if len(all_letters_in_response) == 0:
            return "incorrect"
        elif len(all_letters_in_response) == 1:
            if all_letters_in_response[0] == correct_answer:
                return "correct"
            else:
                return "incorrect"
        else:
            if correct_answer in all_letters_in_response:
                return "under review"
            else:
                return "incorrect"


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
    results = {
        prompt.name: {"correct": 0, "under review": 0, "incorrect": 0, "total": 0}
        if isinstance(prompt, PromptMaker)
        else prompt
        for prompt in prompts
    }
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
                    if isinstance(prompt, Prompt):
                        chosen_prompt = prompt.gen()
                    else:
                        chosen_prompt = prompt
                    start_time = time.time()
                    response = query_model(
                        chosen_prompt,
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
                            "prompt": chosen_prompt,
                            "question": question,
                            "correct_answer": correct_answer,
                            "response": response_dict,
                            "marked_correct": is_correct,
                            "wall_time": wall_time,
                        }
                    )
                    results[chosen_prompt]["total"] += 1
                    if is_correct:
                        results[chosen_prompt]["correct"] += 1
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
                correct_answer = example["answer"]
                choice_A, choice_B, choice_C, choice_D = (
                    example["A"],
                    example["B"],
                    example["C"],
                    example["D"],
                )
                multiple_choice_question = """Question: {question}\nChoices:\n(A) {choice_A}\n(B) {choice_B}\n(C) {choice_C}\n(D) {choice_D}\nAnswer:""".format(
                    question=question,
                    choice_A=choice_A,
                    choice_B=choice_B,
                    choice_C=choice_C,
                    choice_D=choice_D,
                )
                for prompt in prompts:
                    if isinstance(prompt, PromptMaker):
                        if prompt.few_shots:
                            category = example["config"]
                            chosen_prompt = prompt.gen(category)
                        else:
                            chosen_prompt = prompt.gen()
                    else:
                        chosen_prompt = prompt
                    start_time = time.time()
                    response = query_model(
                        chosen_prompt.prompt,
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
                    eval_result = evaluate_mmlu_response(
                        response.choices[0], correct_answer, json_mode
                    )

                    if json_mode:
                        multiple_choice_question = (
                            multiple_choice_question
                            + '\nRemember to return a JSON with the answer to the question containing only the letter "A", "B", "C" or "D", labeled as "answer".'
                        )

                    information["calls"].append(
                        {
                            "prompt_name": prompt.name,
                            "prompt": chosen_prompt.prompt,
                            "question": "Question: "
                            + multiple_choice_question
                            + "\n"
                            + multiple_choice_question
                            if reread
                            else multiple_choice_question,
                            "correct_answer": correct_answer,
                            "response": response_dict,
                            "mark": eval_result,
                            "wall_time": wall_time,
                            "config": example["config"],
                        }
                    )
                    results[prompt.name]["total"] += 1
                    if eval_result == "correct":
                        results[prompt.name]["correct"] += 1
                    elif eval_result == "under review":
                        results[prompt.name]["under review"] += 1
                    elif eval_result == "incorrect":
                        results[prompt.name]["incorrect"] += 1
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
    file_path = f"data/benchmarking/eval_results_{current_datetime}_part_{((count//log_interval))}.json"
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)
    print(f"Written results to {file_path}")


def load_mmlu(configs: List[str], split: str) -> pd.DataFrame:
    column_names = ["input", "A", "B", "C", "D", "answer"]

    # List to store each DataFrame
    dataframes = []

    # Loop through the list of file names
    for file_name in configs:
        # Read the CSV file with specified column names and append to the list
        df = pd.read_csv(
            "data/mmlu/data/" + split + "/" + file_name + "_test.csv",
            names=column_names,
        )
        df["config"] = file_name
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def find_quotes_with_letters(text):
    pattern = r'["\']([A-D])["\']'
    matches = re.findall(pattern, text)
    return matches


def find_parentheses_with_letters(text):
    pattern = r"\(\s*([A-D])\s*\)"
    matches = re.findall(pattern, text)
    return matches


from typing import List
import random


def sample_string(list: List[str]):
    return list[random.randint(0, len(list) - 1)]


class PromptMaker:
    def __init__(
        self,
        name: str,
        vanillas: str or List[str] or None,
        instructions: str or List[str] or None = None,
        spacing: str or None = None,
        few_shots: str or List[str] or None = None,
        randomize_shots: bool = False,
    ):
        self.name = name
        self.vanillas = vanillas
        self.spacing = spacing
        self.instructions = instructions
        self.randomize_shots = randomize_shots
        if few_shots == "MMLU":
            self.few_shots = {
                "STEM": [
                    "Question: A 0.217 g sample of HgO (molar mass = 217 g) reacts with excess iodide ions according to the reaction shown above. Titration of the resulting solution requires how many mL of 0.10 M HCl to reach equivalence point?\nChoices:\n(A) 1.0 mL\n(B) 10 mL\n(C) 20 mL\n(D) 50 mL\nAnswer: (C)",
                    "Question: Many Web browsers allow users to open anonymous windows. During a browsing session in an anonymous window, the browser does not record a browsing history or a list of downloaded files. When the anonymous window is exited, cookies created during the session are deleted. Which of the following statements about browsing sessions in an anonymous window is true?\nChoices:\n(A) The activities of a user browsing in an anonymous window will not be visible to people who monitor the user's network, such as the system administrator.\n(B) Items placed in a Web store's shopping cart for future purchase during the anonymous browsing session will not be saved on the user's computer.\n(C) A user will not be able to log in to e-mail or social media accounts during the anonymous browsing session.\n(D) A user browsing in an anonymous window will be protected from viruses launched from any web sites visited or files downloaded.\nAnswer: (B)",
                    "Question: A point pole has a strength of 4π * 10^-4 weber. The force in newtons on a point pole of 4π * 1.5 * 10^-4 weber placed at a distance of 10 cm from it will be\nChoices:\n(A) 15 N.\n(B) 20 N.\n(C) 7.5 N.\n(D) 3.75 N.\nAnswer: (A)",
                    "Question: Joe was in charge of lights for a dance. The red light blinks every two seconds, the yellow light every three seconds, and the blue light every five seconds. If we include the very beginning and very end of the dance, how many times during a seven minute dance will all the lights come on at the same time? (Assume that all three lights blink simultaneously at the very beginning of the dance.)\nChoices:\n(A) 3\n(B) 5\n(C) 6\n(D) 15\nAnswer: (D)",
                    "Question: The pleura\nChoices:\n(A) have no sensory innervation.\n(B) are separated by a 2 mm space.\n(C) extend into the neck.\n(D) are composed of respiratory epithelium.\nAnswer: (C)",
                ],
                "Humanities": [
                    "Question: Turtles live long lives and are happy creatures, unless they are injured.\nChoices:\n(A) (L • H) ≡ I\n(B) (L • H) ∨ I\n(C) L • (H ∨ I)\n(D) L • (H ⊃ R)\nAnswer: (B)",
                    "Question: A son owed a creditor $5,000. The son's father contacted the creditor and told him that he wanted to pay the son's debt. The father signed a document that stated the father would pay the son's debt at a rate of $500 a month for 10 months. The creditor made no written or oral commitment to forbear to sue the son to collect the $5,000 debt, and the father made no oral or written request for any such forbearance. For the next five months, the father made and the creditor accepted the $500 monthly payments as agreed. During that period, the creditor, in fact, did forbear to take any legal action against the son. However, the father then informed the creditor that he would make no further payments on the debt. Which of the following is the most persuasive argument that the father is liable to the creditor under the terms of their agreement?\nChoices:\n(A) The father's promise and the creditor's reliance thereon, if proved, gave rise to a valid claim by the creditor against the father based on the doctrine of promissory estoppel. \n(B) Because it was foreseeable that the father's promise would induce the creditor to forbear taking any action against the son, such forbearance was, as a matter of law, a bargained-for consideration for the father's promise. \n(C) The father's five payments to the creditor totaling $2,500 manifested a serious intent on the father's part to be contractually bound, and such manifestation is generally recognized as an effective substitute for consideration. \n(D) By assuming the antecedent debt obligation that the son owed to the creditor, the father became a surety whose promise to the creditor was enforceable, since it was in writing and supported by adequate consideration. \nAnswer: (A)",
                    'Question: This question refers to the following information.\n""Society in every state is a blessing, but government even in its best state is but a necessary evil; in its worst state an intolerable one; for when we suffer, or are exposed to the same miseries by a government, which we might expect in a country without government, our calamity is heightened by reflecting that we furnish the means by which we suffer. Government, like dress, is the badge of lost innocence; the palaces of kings are built on the ruins of the bowers of paradise. For were the impulses of conscience clear, uniform, and irresistibly obeyed, man would need no other lawgiver; but that not being the case, he finds it necessary to surrender up a part of his property to furnish means for the protection of the rest; and this he is induced to do by the same prudence which in every other case advises him out of two evils to choose the least. Wherefore, security being the true design and end of government, it unanswerably follows that whatever form thereof appears most likely to ensure it to us, with the least expense and greatest benefit, is preferable to all others.""\nThomas Paine, Common Sense, 1776\nWhich of the following ""miseries"" alluded to above were most condemned by Anti-Federalists of the post-Revolutionary era?\nChoices:\n(A) Organized response to Bacon\'s Rebellion\n(B) Federal response to Shays\'s Rebellion\n(C) Federal response to Pontiac\'s Rebellion\n(D) Federal response to the Whiskey Rebellion\nAnswer: (D)',
                    "Question: Which of the following is true of a valid categorical syllogism?\nChoices:\n(A) The minor premise must deny the antecedent\n(B) The major premise must affirm the consequent\n(C) The middle term must be used in at least one premise in a universal or unqualified sense\n(D) All of the above\nAnswer: (C)",
                    "Question: How can the Upanishads be characterized?\nChoices:\n(A) Ritual texts\n(B) Philosophical texts\n(C) Hymns\n(D) Origin stories\nAnswer: (B)",
                ],
                "Social Sciences": [
                    "Question: Which of the following is not a problem associated with official statistics on strike action?\nChoices:\n(A) most strikes go unnoticed by employers and the mass media\n(B) not all industrial disputes will be reported by the employer\n(C) the definition of strikes excludes those that involve fewer than ten workers or last less than one day\n(D) it is hard to compare strikes that were measured in different ways\nAnswer: (A)",
                    "Question: The realm of policy decisions concerned primarily with relations between the United States and the rest of the world is known as\nChoices:\n(A) terrorism policy.\n(B) economic policy.\n(C) foreign policy.\n(D) international policy.\nAnswer: (C)",
                    "Question: In terms of Hofstede’s (1980) five cultural dimensions, the United States scores at the top of the scale on:\nChoices:\n(A) individualism and power distance.\n(B) individualism.\n(C) power distance and masculinity.\n(D) uncertainty avoidance.\nAnswer: (B)",
                    "Question: For a stationary autoregressive process, shocks will\nChoices:\n(A) Eventually die away\n(B) Persist indefinitely\n(C) Grow exponentially\n(D) Never occur\nAnswer: (A)",
                    "Question: Which of the following statements is NOT accurate regarding the services provided by local governments in the United States?\nChoices:\n(A) Duplication of efforts occurs often.\n(B) Social problems of the central city spill over into the surrounding residential suburbs.\n(C) Inefficiency in providing services occurs often.\n(D) One neighborhood's efforts to reduce pollution are always supported by neighboring communities.\nAnswer: (D)",
                ],
                "Other": [
                    "Question: In contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .\nChoices:\n(A) Buycotts, Boycotts, Blockchain technology, Charitable donations\n(B) Buycotts, Boycotts, Digital technology, Increased Sales\n(C) Boycotts, Buyalls, Blockchain technology, Charitable donations\n(D) Boycotts, Buycotts, Digital technology, Increased Sales\nAnswer: (D)",
                    "Question: In the assessment of the hand function which of the following is true?\nChoices:\n(A) Abduction of the thumb is supplied by spinal root T2\n(B) Opposition of the thumb by opponens policis is supplied by spinal root T1\n(C) Finger adduction is supplied by the median nerve\n(D) Finger abduction is mediated by the palmar interossei\nAnswer: (B)",
                    "Question: What characteristic is not a key feature of the 'open systems' model of management?\nChoices:\n(A) Morale\n(B) Innovation\n(C) Growth resource\n(D) Adaptation\nAnswer: (A)",
                    "Question: When older adults move to a new state after retirement, which of the following is the more likely destination?\nChoices:\n(A) Texas\n(B) California\n(C) Hawaii\n(D) Vermont\nAnswer: (A)",
                    "Question: Which of these songs was a Top 10 hit for the rock band The Police?\nChoices:\n(A) 'Radio Ga-Ga'\n(B) 'Ob-la-di Ob-la-da'\n(C) 'De Do Do Do De Da Da Da'\n(D) 'In-a-Gadda-Da-Vida'\nAnswer: (C)",
                ],
            }
        else:
            self.few_shots = None

    def __str__(self):
        return self.prompt

    def __repr__(self):
        return self.prompt

    def __hash__(self):
        prompt = ""
        for i in self.instructions:
            prompt += i
        for f in self.few_shots:
            prompt += f
        for v in self.vanillas:
            prompt += v
        prompt += self.name
        prompt += str(self.randomize_shots)
        return hash(prompt)

    def __eq__(self, other):
        prompt1 = ""
        for v in self.vanillas:
            for i in self.instructions:
                for f in self.few_shots:
                    prompt1 += i + f + v + self.name + str(self.randomize_shots)
        prompt2 = ""
        for v in other.vanillas:
            for i in other.instructions:
                for f in other.few_shots:
                    prompt2 += i + f + v + other.name + str(other.randomize_shots)
        return prompt1 == prompt2

    def gen(self, category: str or None = None):
        vanilla = sample_string(self.vanillas)
        instruction = sample_string(self.instructions) if self.instructions else None
        space = sample_string(self.spacing) if self.spacing else None
        if self.few_shots:
            shots = self.few_shots[mmlu_split[category]]
            if self.randomize_shots:
                random.shuffle(shots)
        else:
            shots = None
        return Prompt(vanilla, instruction, space, shots)

    def sample_if_needed(prompt_piece: str or List[str] or None):
        if prompt_piece:
            if isinstance(prompt_piece, list):
                return sample_string(prompt_piece)
            else:
                return prompt_piece
        else:
            return ""

    # def format(self, shots):
    #     return '''
    #     {shot1}
    #     {shot2}
    #     {shot3}
    #     {shot4}
    #     {shot5}
    #     '''.format(shot1=shots[0], shot2=shots[1], shot3=shots[2], shot4=shots[3], shot5=shots[4])

    # def space(self):
    #     return sample_string(self.spacing)


class Prompt:
    def __init__(
        self,
        vanilla: str,
        instruction: str or None,
        space: str or None,
        shots: str or None,
    ):
        self.vanilla = vanilla
        self.instruction = instruction
        self.space = space
        self.shots = shots
        self.prompt = self.make_prompt()

    def make_prompt(self):
        if self.shots:
            if self.vanilla and self.instruction:
                return """{vanilla}{space}{instruction}{space}{shot1}{space}{shot2}{space}{shot3}{space}{shot4}{space}{shot5}{space}""".format(
                    vanilla=self.vanilla,
                    instruction=self.instruction,
                    space=self.space,
                    shot1=self.shots[0],
                    shot2=self.shots[1],
                    shot3=self.shots[2],
                    shot4=self.shots[3],
                    shot5=self.shots[4],
                )
            elif self.vanilla and not self.instruction:
                return """{vanilla}{space}{shot1}{space}{shot2}{space}{shot3}{space}{shot4}{space}{shot5}{space}""".format(
                    vanilla=self.vanilla,
                    instruction=self.instruction,
                    space=self.space,
                    shot1=self.shots[0],
                    shot2=self.shots[1],
                    shot3=self.shots[2],
                    shot4=self.shots[3],
                    shot5=self.shots[4],
                )
        elif self.instruction and self.vanilla:
            return "{vanilla}{space}{instruction}{space}".format(
                vanilla=self.vanilla, instruction=self.instruction, space=self.space
            )
        else:
            return self.vanilla

    def __str__(self):
        return self.prompt

    def __repr__(self):
        return self.prompt

    def __hash__(self):
        return hash(self.prompt)

    def __eq__(self, other):
        return self.prompt == other.prompt

    def gen(self):
        return self.prompt


mmlu_split = {
    "high_school_european_history": "Humanities",
    "business_ethics": "Other",
    "clinical_knowledge": "Other",
    "medical_genetics": "Other",
    "high_school_us_history": "Humanities",
    "high_school_physics": "STEM",
    "high_school_world_history": "Humanities",
    "virology": "Other",
    "high_school_microeconomics": "Social Sciences",
    "econometrics": "Social Sciences",
    "college_computer_science": "STEM",
    "high_school_biology": "STEM",
    "abstract_algebra": "STEM",
    "professional_accounting": "Other",
    "philosophy": "Humanities",
    "professional_medicine": "Other",
    "nutrition": "Other",
    "global_facts": "Other",
    "machine_learning": "STEM",
    "security_studies": "Social Sciences",
    "public_relations": "Social Sciences",
    "professional_psychology": "Social Sciences",
    "prehistory": "Humanities",
    "anatomy": "STEM",
    "college_medicine": "Other",
    "high_school_government_and_politics": "Social Sciences",
    "college_chemistry": "STEM",
    "logical_fallacies": "Humanities",
    "high_school_geography": "Social Sciences",
    "elementary_mathematics": "STEM",
    "human_aging": "Other",
    "college_mathematics": "STEM",
    "high_school_psychology": "Social Sciences",
    "formal_logic": "Humanities",
    "high_school_statistics": "STEM",
    "international_law": "Humanities",
    "high_school_mathematics": "STEM",
    "high_school_computer_science": "STEM",
    "conceptual_physics": "STEM",
    "miscellaneous": "Other",
    "high_school_chemistry": "STEM",
    "marketing": "Other",
    "professional_law": "Humanities",
    "management": "Other",
    "college_physics": "STEM",
    "jurisprudence": "Humanities",
    "world_religions": "Humanities",
    "sociology": "Social Sciences",
    "us_foreign_policy": "Social Sciences",
    "high_school_macroeconomics": "Social Sciences",
    "computer_security": "STEM",
    "moral_scenarios": "Humanities",
    "moral_disputes": "Humanities",
    "electrical_engineering": "STEM",
    "astronomy": "STEM",
    "college_biology": "STEM",
}
