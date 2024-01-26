import os

import json

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import MetadataMode
import re
import uuid

from llama_index.llms import OpenAI
from tqdm.notebook import tqdm


QUERIES_FPATH = './data/processed/queries.json'
RELEVANT_DOCS_FPATH = './data/processed/relevant_docs.json'
ANSWERS_FPATH = './data/processed/answers.json'
CORPUS_FPATH = './data/processed/corpus.json'
DATASET_FPATH = './data/processed/dataset.json'


def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f'Loaded {len(docs)} docs')
    
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f'Parsed {len(nodes)} nodes')

    corpus = {node.node_id: node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes}
    return corpus



###### Generate queries and answers #####
def generate_queries_and_answers(
    corpus,
    num_questions_per_chunk=2,
    prompt_template=None,
    verbose=False,
):
    """
    Automatically generate hypothetical questions and answers that could be 
    answered with the doc in the corpus.
    """
    llm = OpenAI(model='gpt-3.5-turbo')

    prompt_template = prompt_template or """\
    Context information is provided below.

    ---------------------
    {context_str}
    ---------------------

    With the provided context information and no prior knowledge,
    create {num_questions_per_chunk} question(s) and their corresponding answer(s) 
    for an upcoming quiz/examination. Answers should be concise, limited to 1-5 words. 
    The questions and answers should be diverse in nature across the document 
    and directly related to the context information."
"""


    queries = {}
    answers = {}
    relevant_docs = {}
    for node_id, text in tqdm(corpus.items()):
        query = prompt_template.format(context_str=text, num_questions_per_chunk=num_questions_per_chunk)
        response = llm.complete(query)
 
        result = str(response).strip().split("\n")
        q_a_pairs = zip(result[0::2], result[1::2])  # Assuming alternating questions and answers

        for question, answer in q_a_pairs:
            question = re.sub(r"^\d+[\).\s]", "", question).strip()
            if len(question) > 0 and len(answer) > 0:
                question_id = str(uuid.uuid4())
                question = question.replace("Question:", "").strip()
                queries[question_id] = question
                answer = answer.replace("Answer:", "").strip()
                answers[question_id] = answer
                relevant_docs[question_id] = [node_id]
    
    return queries, answers, relevant_docs



if __name__ == "__main__":
    FILES = ['./data/raw/test.txt']
    corpus = load_corpus(FILES, verbose=True)
    
    if not os.path.exists('./data'):
       os.mkdir('./data')
       
    
    with open(CORPUS_FPATH, 'w') as f:
      json.dump(corpus, f)
      
    with open(CORPUS_FPATH, 'r+') as f:
        corpus = json.load(f)

    # Select only 5 examples for training
    corpus = {k: corpus[k] for k in list(corpus.keys())}


    train_queries, train_answers, train_relevant_docs = generate_queries_and_answers(
    corpus,
    num_questions_per_chunk=2,
    verbose=True)

    with open(QUERIES_FPATH, 'w+') as f:
        json.dump(train_queries, f)

    with open(ANSWERS_FPATH, 'w+') as f:
        json.dump(train_answers, f)

    with open(RELEVANT_DOCS_FPATH, 'w+') as f:
        json.dump(train_relevant_docs, f)    

    train_dataset = {
        'queries': train_queries,
        'answers': train_answers,
        'corpus': corpus,
        'relevant_docs': train_relevant_docs,
    }

    if os.path.exists(DATASET_FPATH):
        os.remove(DATASET_FPATH)
    with open(DATASET_FPATH, 'w+') as f:
        json.dump(train_dataset, f)

    
    


