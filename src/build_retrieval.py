from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import pandas as pd
import os
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from utils import extract_main_topics
import chromadb

PATH_FILE = 'data/raw/test.parquet'

def generate_text_file(path_file):
    """ 
    Generate a text file from the parquet file
    """
    df = pd.read_parquet(path_file)
    text = ''
    for line in df['text']:
        text += line

    # Extract main topics
    main_topics = extract_main_topics(text)

    # Save as txt file
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')

    with open('data/processed/main_topics.txt', 'w') as f:
        for topic in main_topics:
            f.write(topic + '\n')

    # Save as txt file
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')
    with open('data/raw/train.txt', 'w') as f:
        f.write(text)

    return text
        

def get_token_splits(text):
    """
    Splits the given text into tokenized chunks.

    Args:
        text (str): The input text to be split.

    Returns:
        list: A list of tokenized chunks.
    """
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=256,
        chunk_overlap=0)

    character_split_texts = character_splitter.split_text(text)

    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts



def init_retrieval_model(token_split_texts, embedding_function):
    """ 
    Initialize the retrieval model
    
    Args:
        token_split_texts (list): A list of tokenized and split texts.
        embedding_function (function): The embedding function to be used for encoding the texts.
    
    Returns:
        None
    """
    chroma_client = chromadb.PersistentClient("chroma.db")
    chroma_collection = chroma_client.get_or_create_collection("test", embedding_function=embedding_function)
    ids = [str(i) for i in range(len(token_split_texts))]
    chroma_collection.add(ids=ids, documents=token_split_texts)

    print("Collection name:", chroma_collection.name)


def main():
    text = generate_text_file(PATH_FILE)
    
    token_split_texts = get_token_splits(text)
    
    embedding_function = SentenceTransformerEmbeddingFunction()
    init_retrieval_model(token_split_texts, embedding_function)

    print("Done!")


if __name__ == "__main__":
    main()