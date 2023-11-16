from role_prompting import query_model, evaluate_response, evaluate_prompts

key = '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'

prompts = [
    "As an expert mathematician, solve the following and indicate your final answer with '####': ",
    "Solving step by step, then write your final answer after '####': ",
    "As a math teacher, explain and conclude your final answer with '####': ",
]

dataset = 'gsm8k'

split = 'test'

output = query_model(key, prompts[0], '\nWhat is 2+2?', 'davinci')
print(output)