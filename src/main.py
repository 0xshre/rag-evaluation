EXIT_PROMPT = "exit"

import dspy
from chromadb_rm import ChromadbRM
import os

class GenerateAnswer(dspy.Signature):
    """Answer questions given the context"""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Answer the question in 1-5 words")

class RAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


def setup():
    """
    Setup the dsypy and retrieval models
    """

    turbo = dspy.OpenAI(model='gpt-3.5-turbo')

    chroma_rm = ChromadbRM(collection_name="test", persist_directory="chroma.db", local_embed_model="sentence-transformers/paraphrase-MiniLM-L6-v2",
                                   openai_api_key = os.environ["OPENAI_API_KEY"])

    dspy.settings.configure(lm=turbo, rm=chroma_rm)
    
    rag = RAG()

    return rag

if __name__ == "__main__":
    
    rag = setup()

    while True:
        print(f"\n\nEnter the prompt or type {EXIT_PROMPT} to exit\n")
        # Get the prompt
        prompt = input()
        # Check if the user wants to exit
        if prompt == EXIT_PROMPT:
            break
        
        # Get the response
        response = rag(prompt)

        # Print the response
        print(f"\n\nAnswer: {response.answer}")