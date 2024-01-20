# To-do List

Now that we have  a working prototype, the next steps are as follows:

- Connect to a self hosted LLM
- Set up an evaluation pipeline
    - Fa
- host the vector database as an API


- [Evaluating the Ideal Chunk Size for a RAG System using LlamaIndex](https://blog.llamaindex.ai/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5)




Sudhanva's to do
---

- start writing test cases for inputs and non-inputs
- personal recommendation is to use langchain + llama2 7b + transformers + chroma
- fine tune llama with transformers on HF on this data
- chroma will store your vectors, make them searchable with ranking
- langchain will pull from chroma and you have to build proper chain and connect to llama (most challenging part)
- ethics/illegal input considerations
- observability cosniderations



-----

Inferece
---
https://huggingface.co/docs/text-generation-inference/index
https://modal.com/docs/examples/text_generation_inference



RAG Evaluation
---
https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo.html
https://zilliz.com/blog/how-to-evaluate-retrieval-augmented-generation-rag-applications
https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a

**GOLD STANDARD DATASET**

We first need to create a dataset that is the gold standard for RAG evaluation. This dataset will be used to evaluate the performance of the RAG system.

Gold standard dataset contains the following:

- Query or Prompt
- Answer
- Context


Each query can be answered by the context provided. The answer is the ground truth answer to the query.
(Context -> Question -> Answer)


**METRICS**

To evaluate the retrieval performance of the RAG system, we use the following metrics:

FULL EVALUATION

- [Faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html): Measures the factual consistency of the generated answer against the given context
- [Answer Relevance](https://docs.ragas.io/en/latest/concepts/metrics/answer-relevance.html):  Assesses how pertinent the generated answer is to the given prompt

BINARY EVALUATION

- [Answer Correctness](https://docs.ragas.io/en/latest/concepts/metrics/answer-correctness.html): Measures the correctness of the generated answer against the ground truth answer

RETRIEVAL EVALUATION

- Context Precision is a metric that evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not.
- Context Recall measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth


**RETRIEVAL EVALUATION**

This section describes how we evaluate the retrieval performance of the RAG system. We use the gold standard dataset to evaluate the retrieval performance of the RAG system.

1. For every query, we first generate the answer withholding the gold standard answer.  
(Context + Question -> Answer). This provides us with an upper bound on the retrieval performance of the RAG system. Note that context here does not translate to the chunked documents. It is the original context document.

    - Metrics: Answer Correctness

2. Now, we use the similarity search to retrieve the top k contexts for the query. We then generate the answer using the top k contexts. (Context + Question -> Answer). This provides us with the retrieval performance of the RAG system.

Evaluation exampe




COMMENTS
---

Why we are not using ranking metrics?

- Since, we are creating the sythetic dataset from only one context document, we do not have multiple contexts to rank. Hence, we are not using ranking metrics. This is under the assumption that question cannot be answered by more than one context document. Although, this is not true in general, we are making this assumption for the sake of simplicity. For now, we will use binary metrics to evaluate the retrieval performance of the RAG system.

No Indexing Metrics
No Cross Encoding / Reranking
No Hybrid Search
No Meta Data: After we have the top k contexts, we can use the meta data to rank the contexts. This is not implemented yet.
Distraction due to irrelevant contexts



FUTURE WORK
---
- Lost in the middle evaluation
- Approximate Nearest Neighbor Errors

More links
---
1. [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://arxiv.org/pdf/2311.09476.pdf)
2. [Tutorial of RAG - Google Colab](https://medium.com/@murtuza753/using-llama-2-0-faiss-and-langchain-for-question-answering-on-your-own-data-682241488476)
