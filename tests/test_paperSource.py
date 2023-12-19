from prompt_systematic_review.paperSource import Paper


def testAll():
    # 1st test on Paper
    authorss = [
        "Reid Pryzant",
        "Dan Iter",
        "Jerry Li",
        "Yin Tat Lee",
        "Chenguang Zhu",
        "Michael Zeng",
    ]
    keywordss = ["prompt optimization", "prompt engineering", "few-shot learning"]

    paper = Paper(
        "Automatic Prompt Optimization with Gradient Descent and Beam Search",
        authorss,
        "https://arxiv.org/abs/2305.03495",
        2023,
        keywordss,
        "Large Language Models (LLMs) have shown impressive performance as general purpose agents, but their abilities remain highly dependent on prompts which are hand written with onerous trial-and-error effort. We propose a simple and nonparametric solution to this problem, Automatic Prompt Optimization (APO), which is inspired by numerical gradient descent to automatically improve prompts, assuming access to training data and an LLM API. The algorithm uses minibatches of data to form natural language gradients that criticize the current prompt. The gradients are then propagated into the prompt by editing the prompt in the opposite semantic direction of the gradient. These gradient descent steps are guided by a beam search and bandit selection procedure which significantly improves algorithmic efficiency. Preliminary results across three benchmark NLP tasks and the novel problem of LLM jailbreak detection suggest that Automatic Prompt Optimization can outperform prior prompt editing techniques and improve an initial prompt's performance by up to 31%, by using data to rewrite vague task descriptions into more precise annotation instructions.",
        "23993",
    )

    # Ensures that the title + authors method is working and that the keywords are matching
    assert (
        paper.__str__()
        == "Automatic Prompt Optimization with Gradient Descent and Beam Search, by Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, Michael Zeng"
    )
    assert paper.matchingKeyWords() == keywordss

    # Ensures all the proper headings are in the paper
    paperDict = paper.__dict__
    assert (
        paperDict["title"]
        == "Automatic Prompt Optimization with Gradient Descent and Beam Search"
    )
    assert paperDict["authors"] == authorss
    assert paperDict["url"] == "https://arxiv.org/abs/2305.03495"
    assert paperDict["dateSubmitted"] == 2023
    assert (
        paperDict["abstract"]
        == "Large Language Models (LLMs) have shown impressive performance as general purpose agents, but their abilities remain highly dependent on prompts which are hand written with onerous trial-and-error effort. We propose a simple and nonparametric solution to this problem, Automatic Prompt Optimization (APO), which is inspired by numerical gradient descent to automatically improve prompts, assuming access to training data and an LLM API. The algorithm uses minibatches of data to form natural language gradients that criticize the current prompt. The gradients are then propagated into the prompt by editing the prompt in the opposite semantic direction of the gradient. These gradient descent steps are guided by a beam search and bandit selection procedure which significantly improves algorithmic efficiency. Preliminary results across three benchmark NLP tasks and the novel problem of LLM jailbreak detection suggest that Automatic Prompt Optimization can outperform prior prompt editing techniques and improve an initial prompt's performance by up to 31%, by using data to rewrite vague task descriptions into more precise annotation instructions."
    )
    assert paperDict["paperId"] == "23993"

    # Creates other paper with same title but different information to test out eq method in paperSource
    otherPaper = Paper(
        "Automatic Prompt Optimization with Gradient Descent and Beam Search",
        ["Alexander Boyle", "Cassandra Darcy"],
        "https://arxiv.org/abs/2305.03495",
        1400,
        ["role prompting", "topic modeling"],
        "Different Absstract",
        "",
    )
    assert paper.__eq__(otherPaper)
