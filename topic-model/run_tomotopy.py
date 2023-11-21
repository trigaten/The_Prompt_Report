#%% imports
import argparse
import html

import tomotopy as tp
import pandas as pd
import numpy as np


#%% arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_fname", type=str, default="processed/train.metadata.jsonl")
parser.add_argument("--num_topics", type=int, default=50)
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--top_words_to_display", type=int, default=15)
parser.add_argument("--top_docs_to_display", type=int, default=5)

args = parser.parse_args()

# %% load data
data = pd.read_json(args.input_fname, lines=True)

lda = tp.LDAModel(k=args.num_topics, seed=42)
for doc in data["tokenized_text"]:
    lda.add_doc(doc.split())

# %% train model
for i in range(0, args.iterations, 10):
    lda.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, lda.ll_per_word))

#%% print out a summary
print(lda.summary())

# %% create the html template we will populate
html_template = '''  
<!DOCTYPE html>  
<html lang="en">  
<head>  
    <meta charset="UTF-8">  
    <meta name="viewport" content="width=device-width, initial-scale=1.0">  
    <title>Topics and Documents</title>  
    <style>  
        body {{  
            font-family: Arial, sans-serif;  
        }}  
        .topic-container {{  
            margin-bottom: 20px;  
            border-bottom: 1px solid #ccc;  
            padding-bottom: 10px;  
        }}  
        .topic {{  
            display: flex;  
            align-items: center;  
        }}  
        .topic-checkbox {{  
            margin-right: 10px;  
        }}  
        details {{  
            margin-left: 30px;  
        }}  
        summary {{  
            cursor: pointer;  
        }}  
        ul {{  
            list-style-type: none;  
            padding-left: 0;  
        }}  
        li {{  
            background-color: #f0f0f0;  
            border: 1px solid #ccc;  
            border-radius: 4px;  
            padding: 5px;  
            margin-bottom: 5px;  
        }}  
    </style>  
</head>  
<body>  
    <div id="content">  
        {content}  
    </div>  
</body>  
</html>  
'''

#%% create the html content
def retrieve_top_docs_for_topic(theta, docs, topic_idx, n=3, max_chars=250, display=True):
    """
    Show the top docs per topic
    """
    theta_sorted = (-theta[:, topic_idx]).argsort()[:n]
    doc_topics_i = []
    for doc_idx in theta_sorted:
        doc = docs[doc_idx]
        if display:
            print(">>", doc[:max_chars].strip(), "\n")
        doc_topics_i.append(doc)
    return doc_topics_i


#%% create the document-topic distribution matrix
theta = np.array([doc.get_topic_dist() for doc in lda.docs])
docs = data.to_dict(orient="records")

# %% create the html content
content = []
# for each topic, get the top words and top documents
for topic_idx in range(args.num_topics):
    # get the top words
    topic = " ".join([word for word, _ in lda.get_topic_words(topic_idx, top_n=args.top_words_to_display)])

    topic_html = f'<div class="topic-container">'  
    topic_html += f'<div class="topic"><input type="checkbox" class="topic-checkbox">{topic_idx}: {topic}</div>'  
    topic_html += f'<details><summary>Documents</summary><ul>'  
    
    # get the top documents
    for rank, doc in enumerate(retrieve_top_docs_for_topic(theta, docs, topic_idx, n=args.top_docs_to_display, display=False)):  
        # get the title, abstract, and url; format as html
        title, abstract, url = doc["title"], doc["abstract"], doc["url"]
        topic_html += f'<li><a href={url}>{html.escape(title)}</a></li>'
        # also optionally expand to include the abstract
        topic_html += f'<details><summary>Abstract</summary><p>{html.escape(abstract)}</p></details>'
      
    topic_html += f'</ul></details></div>'  
    content.append(topic_html)  

review_html = html_template.format(content=''.join(content))   

#%% write the html to a file  
with open(f'topic_outputs-{args.num_topics}.html', 'w') as f:  
    f.write(review_html)