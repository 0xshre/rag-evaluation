QA RAG (Retrieval-Augmented Generation) System with Evaluation
===

## Setup

1. Clone the repository

```bash
git clone [repository URL]
```

2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate 
```

3. Install the requirements

```bash
pip install -r requirements.txt
```

4. Download the data

```bash
bash setup_script.sh
```

> This will download the data and place it in the correct directories.


## Usage

To build the chromadb, run the following command:

```bash
python src/build_retriever.py
```

This will build the database and place it in the `chroma.db` directory. This automatically uses the `data/raw/` directory to build the embeddings.

Then, use the following command to run the entire pipeline as a command line interface:

```bash
python src/main.py
```


## Report

The evaluation can be found in the following link:

https://api.wandb.ai/links/prasadshreyas/mrby6yi9



## Methodology

### 1. Data Collection
- Dataset: `wikitext-raw-2` 
- Chunking: `RecursiveCharacterTextSplitter`
- Sentence Embedding: `sentence-transformers/paraphrase-MiniLM-L6-v2` (ndim: 384)


### 2. Generating Synthetic Questions and Answers

Inspired by HotpotQA, I used the following prompt to generate questions and answers for the dataset.

Using the following prompt:

```markdown
Context information is provided below.

---------------------
{context_str}
---------------------

With the provided context information and no prior knowledge,
create {num_questions_per_chunk} question(s) and their corresponding answer(s) 
for an upcoming quiz/examination. Answers should be concise, limited to 1-5 words. 
The questions and answers should be diverse in nature across the document 
and directly related to the context information."
```

- Total number of questions generated: 427


Sample questions and answers:

```markdown
Q: Who directed the play "How to Curse" in 2007?
A: Josie Rourke
```

### Setting up the RAG

For retrieval-augmented generation, I used DSPy with the RM set to `chromadb` and the LM configured as `gpt-3.5-turbo.` 

```python
class GenerateAnswer(dspy.Signature):
"""Answer questions with short factoid answers."""

context = dspy.InputField(desc="may contain relevant facts")
question = dspy.InputField()
answer = dspy.OutputField(desc="Explain with words between 1 and 5 words")

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
```

**NOTE**: DSPy package did not support local embeddings in chromadb which led me to add that feature. A PR has been submitted. can be found here:
https://github.com/stanfordnlp/dspy/pull/319



### Experiments

The results of the experiments are available in the following link:
https://api.wandb.ai/links/prasadshreyas/mrby6yi9


Before we proceed with the experiments, let's define the metrics we will be using to evaluate the performance of the RAG.

**Metrics**

[RAGAS](https://docs.ragas.io/en/latest/concepts/metrics/) to evaluate the performance of the RAG.

- Faithfulness
- Answer Relevance
- Context Precision
- Context Relevancy
- Context Recall
- Answer Semantic Similarity
- **Answer Correctness**


#### 1. Memorization check without Retrieval


```python
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Define the predictor.
generate_answer = dspy.Predict(BasicQA)
```

The main goal of this experiment is to check if the model has memorized the data. For every Q-A pair, we will check the similarity and the correctness of the answer against the ground truth. Note that there is no retrieval involved in this experiment.


| Metric              | Value    |
|---------------------|----------|
| answer_similarity   | 0.85443  |
| answer_correctness  | **0.34959**  |



#### System's performance on synthetic data with ground truth as context
    
Main goal: To check if the model is able to retrieve the correct answer from the ground truth context, check hallucination and correctness of the answer.


| Metric              | Value    |
|---------------------|----------|
| answer_relevancy    | 0.92671  |
| answer_similarity   | 0.99999  |
| context_precision   | 0.44262  |
| context_recall      | 0.94999  |
| context_relevancy   | 0.07294  |
| faithfulness        | 0.8716   |
| answer_correctness  | **0.6250**|

Ended up using a 1024 chunk size hence we see the precision and relevancy being low, while maintaining a high recall because using this is what the question answering system will be using. This is the upper limit of the performance of the pipeline.

#### 3. Hyper-parameter Search

All the experiments we will only use a subset of the questions that have answer correctness are near 0 in the memorization step.

- total_no_of_questions: 427
- test-train split: 0.2
- no_of_questions in test: 86


| Parameter    | Values                          |
|--------------|---------------------------------|
| chunk_size   | 256                             |
| overlap      | 0, 32, 64                       |
| type         | "SIMPLE" (ZERO SHOT), "COMPILED" (FEW SHOT) |
| num_passages | 3, 5                            |


---


#### 4. Ranking Retrieved Passages using Multiple Queries

Steps:
1. First generate $m$ augmented queries for all $n$ queries in the test dataset
2. Then, for each query, find the $k$ number of passages that are most similar to the query
3. Maintain a most_common counter for all the retrieved passages
4. Finally, for each query, find the most common $k$ passages and use them as context for the query
5. Then, use the context to find the most similar answer to the query

```python

from collections import Counter, defaultdict

class GenerateAnswer(dspy.Signature):
    """Answer questions given the context"""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Short factual answer to the question. 1 - 5 words long.")


class QueryReranker(dspy.Signature):
    """Generates 2 more questions to help find the information needed"""
    original_question = dspy.InputField(desc="The original question")
    questions = dspy.OutputField(desc="The question(s) in a new line")

class QueryExpansionRAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.num_passages = num_passages
        self.generate_questions = dspy.ChainOfThought(QueryReranker)
        self.retrieve = dspy.Retrieve(k=self.num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        
    
    def forward(self, question):
        prediction = self.generate_questions(original_question=question)
        questions = prediction.questions.split('\n')
        questions.append(question)
        counter = Counter(defaultdict(int))
        for idx, q in enumerate(questions):
            # Retrieve the top k passages for each question
            passages = self.retrieve(q).passages
            # Count the number of times each passage is retrieved
            for p in passages:
                counter[p] += 1
        context = [p for p, c in counter.most_common(self.num_passages)]
        answer_prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(
            original_question=question,
            questions=questions, context=context, answer=answer_prediction.answer)
```


#### 5. Query Expansion

- Question + Hypothetical Answer -> Retrieved Context 
- Context + Question -> Answer

```python

from collections import Counter, defaultdict

class GenerateAnswer(dspy.Signature):
    """Answer questions given the context"""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Short factual answer to the question. 1 - 5 words long.")


class HypotheticalAnswer(dspy.Signature):
    """Provides a hypothetical question to ask given the question."""
    original_question = dspy.InputField(desc="The question")
    hyp_answer = dspy.OutputField(desc="The hypothetical answer to the question.")

class QueryExpansionRAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.num_passages = num_passages
        self.generate_hypo_answers = dspy.ChainOfThought(HypotheticalAnswer)
        self.retrieve = dspy.Retrieve(k=self.num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        
    
    def forward(self, question):
        prediction = self.generate_hypo_answers(question=question)
        hyp_answer = prediction.hyp_answer
        # combine hyp_answer with the original question
        question = f"{question} {hyp_answer}"
        # retrieve relevant passages
        passages = self.retrieve(question).passages
        # generate answer
        answer_prediction = self.generate_answer(context=passages, question=question)
        return dspy.Prediction(
            original_question=question, context=passages, answer=answer_prediction.answer)
```