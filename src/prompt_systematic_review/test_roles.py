from role_prompting import query_model, evaluate_response, evaluate_prompts

key = "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"

prompts = [
    "As an expert mathematician, solve the following and put four hashtags: '####' before your final answer. ",
    "Solving step by step, then write your final answer before '####': ",
    "As a math teacher, explain and conclude your final answer with '####': ",
]

dataset = "gsm8k"

split = "test"

model = "gpt-3.5-turbo-1106"

output = query_model(key, prompts[0], "\nWhat is 2+2?", model_name=model)
print(output)
