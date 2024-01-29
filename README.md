# Wikipedia RAG using DSPy + ChromaDB

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
