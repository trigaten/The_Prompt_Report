from . import count_models
from . import count_tool_mentions
from . import eval_prompts
from . import evaluate_human_agreement
from . import graph_dataset_citations
from . import graph_models
from . import graph_tool_mentions
from . import keyword_wordcloud
from . import papers_over_time
from . import visualize_authors
from . import graph_gpt_4_benchmarks200
from . import graph_gpt_3_5_benchmarks
from . import run_tomotopy
from . import topicgpt
from . import download_mmlu
from . import graph_internal_references
from . import graph

experiments = [
    count_tool_mentions.Experiment,
    eval_prompts.Experiment,
    evaluate_human_agreement.Experiment,
    graph_dataset_citations.Experiment,
    graph_models.Experiment,
    graph_tool_mentions.Experiment,
    keyword_wordcloud.Experiment,
    papers_over_time.Experiment,
    visualize_authors.Experiment,
    graph_gpt_4_benchmarks200.Experiment,
    graph_gpt_3_5_benchmarks.Experiment,
    run_tomotopy.Experiment,
    topicgpt.Experiment,
    count_models.Experiment,
]
