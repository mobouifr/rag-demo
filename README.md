<div align="center">

# Rag-demo

**A fully local Retrieval Augmented Generation system — no API keys, no cloud, no black boxes.**

*From raw text to semantic search and grounded answers — everything runs on your machine.*

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge)](https://trychroma.com)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge)](https://langchain.com)

</div>

---

## What is this?

`rag-demo` is a working implementation of a RAG (Retrieval Augmented Generation) pipeline built from scratch — no hosted APIs, no paid services. You give it a document. You ask questions. It finds the relevant parts and answers you using a local LLM.

The system is built to be readable and explainable — every component is visible, every step is logged, and the architecture maps directly to how production RAG systems work in the real world.

> The goal is not just to make it work — it is to understand what every layer is actually doing.

---

## How it works

```
Document
  → split into overlapping chunks
  → each chunk converted to a vector (embedding)
  → vectors stored in ChromaDB

Question
  → converted to a vector
  → ChromaDB finds the 3 most similar chunks
  → chunks injected into a prompt as context
  → Ollama (Llama 3.2 3B) generates a grounded answer
  → answer + sources returned
```

The key insight: similar meaning produces similar vectors. Finding relevant chunks is a geometry problem — not keyword matching.

---

## Stack

| Layer | Technology | Role |
|---|---|---|
| Language model | Llama 3.2 3B via Ollama | Generates answers from retrieved context |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) | Converts text to 384-dimensional vectors |
| Vector database | ChromaDB | Stores embeddings, runs similarity search |
| Pipeline | LangChain 1.x (langchain-chroma, langchain-ollama, langchain-huggingface) | Connects all components |
| Runtime | Python 3.12 | Execution environment |

---

## Before you start

> **This runs entirely on CPU. Expect 1–2 minutes per answer.**

No GPU is required. On a machine without NVIDIA CUDA, the LLM runs on CPU at roughly 4 tokens per second. The first question also loads the model into memory — that adds another 30–60 seconds on first run.

This is not a bug. It is the expected behavior of a 3B parameter model on CPU.

---

## Requirements

- Python 3.10+
- 6 GB RAM available
- 3 GB disk space (model weights)
- 100 MB disk space (embedding model, downloaded on first run)
- Linux or macOS — Windows users: use WSL2

---

## Setup

### Automated

```bash
chmod +x setup.sh
./setup.sh
```

Then run:

```bash
source venv/bin/activate
python3 rag.py 2>/dev/null
```

### Manual

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama and pull the model
ollama serve &
ollama pull llama3.2:3b
# Note: Ollama must be running on localhost:11434 before starting the app

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run
python3 rag.py 2>/dev/null
```

---

## Running the app

```bash
# Clean output — recommended for demos
python3 rag.py 2>/dev/null

# Verbose output — shows full Ollama internals, useful for debugging
python3 rag.py
```

The verbose mode prints everything happening inside the LLM server — model loading, memory allocation, token generation speed, context cache. Useful when something breaks. Noisy otherwise.

---

## Example session

```
You: What is RAG and how does it work?

Searching knowledge base...

Answer:
RAG (Retrieval Augmented Generation) is a technique that enhances LLM
responses by providing relevant context retrieved from a knowledge base.

The pipeline has two phases:
1. Indexing — split documents into chunks, embed them, store in a vector DB
2. Query — embed the question, retrieve similar chunks, generate a grounded answer

RAG solves the context window limitation by grounding LLM responses in
specific documents rather than relying on training data alone.

Sources used:
  [1] RAG solves the context window limitation and grounds LLM responses...
  [2] The RAG pipeline has two phases: Indexing phase (done once)...
  [3] Finding the most similar chunk vectors in the database...
```

---

## Using your own document

Open `rag.py` and replace the `DOCUMENT` variable:

```python
DOCUMENT = """
Your content here.
Works in Arabic, French, English, or mixed.
"""
```

To load from a file:

```python
with open("your_document.txt", "r", encoding="utf-8") as f:
    DOCUMENT = f.read()
```

---

## Configuration

All tunable parameters are at the top of `rag.py`:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # embedding model — runs locally
LLM_MODEL       = "llama3.2:3b"        # language model via Ollama
CHUNK_SIZE      = 500                   # characters per chunk
CHUNK_OVERLAP   = 50                    # overlap between adjacent chunks
N_RESULTS       = 3                     # chunks retrieved per question
```

LLM generation behavior is controlled by these parameters in `OllamaLLM`:

```python
temperature=0.1      # lower = more deterministic, higher = more creative
num_ctx=4096         # context window size in tokens
top_p=0.9            # nucleus sampling — controls diversity
top_k=40             # only consider top K tokens by probability
repeat_penalty=1.1   # penalizes repetition in generated text
```

---

## Project structure

```
rag-demo/
├── rag.py            # full pipeline — split, embed, retrieve, generate
├── setup.sh          # automated setup for any machine
├── requirements.txt  # Python dependencies (minimum versions)
├── .gitignore
└── README.md
```

---

## Stage breakdown

<details>
<summary>Indexing — building the knowledge base</summary>
<br/>

When the app starts, it processes the document once before accepting any questions.

**Chunking** — the document is split into overlapping pieces of ~500 characters. Overlap ensures no context is lost at boundaries between chunks.

**Embedding** — each chunk is passed through `all-MiniLM-L6-v2`, a sentence-transformer model that converts text into a 384-dimensional vector. Similar meaning produces similar vectors.

**Storage** — each chunk and its vector are stored in ChromaDB, an in-memory vector database that supports nearest-neighbor search.

</details>

<details>
<summary>Retrieval — finding relevant context</summary>
<br/>

When a question arrives:

1. The question is converted to a 384-dimensional vector using the same embedding model
2. ChromaDB computes the cosine similarity between the question vector and every stored chunk vector
3. The 3 most similar chunks are returned

This is semantic search — it finds relevant chunks based on meaning, not on keyword overlap. "How do I pay?" and "What is the invoice deadline?" are semantically close even though they share no words.

</details>

<details>
<summary>Generation — producing the answer</summary>
<br/>

The retrieved chunks are injected into a prompt template:

```
You are an AI assistant. Answer using only the context below.
If the answer is not in the context, say so honestly.

Context:
[retrieved chunks]

Question:
[user question]

Answer:
```

This prompt is sent to `llama3.2:3b` via the Ollama HTTP API at `localhost:11434`. The model generates text from that exact position, grounded in the provided context.

Temperature is set to `0.1` — close to deterministic, favoring factual consistency over creativity.

</details>

---

## Troubleshooting

**Ollama not responding**

```bash
ollama serve &
curl http://localhost:11434    # should return "Ollama is running"
```

**Model not found**

```bash
ollama list                   # check what is downloaded
ollama pull llama3.2:3b       # re-download if missing
```

**Output is very noisy**

```bash
python3 rag.py 2>/dev/null    # redirect stderr to suppress Ollama logs
```

**PyTorch hanging on startup**

The `CUDA_VISIBLE_DEVICES=""` line at the top of `rag.py` prevents PyTorch from looking for GPU hardware. If you still see a hang:

```bash
export CUDA_VISIBLE_DEVICES=""
python3 rag.py 2>/dev/null
```

**Slow responses**

Expected on CPU. The model generates roughly 4 tokens per second without a GPU. A 200-token answer takes around 50 seconds. For faster inference, you need an NVIDIA GPU with CUDA support.

**venv not active after reopening terminal**

```bash
source venv/bin/activate      # run this every time you open a new terminal
```

---

## Key concepts

<details>
<summary>What is an embedding?</summary>
<br/>

An embedding is a list of numbers that encodes the meaning of a piece of text. The model that produces embeddings is trained so that texts with similar meaning produce similar lists of numbers.

This makes it possible to do arithmetic on meaning. The distance between two vectors corresponds to the semantic distance between the texts they represent. Nearest-neighbor search in vector space is semantic search.

</details>

<details>
<summary>Why chunking with overlap?</summary>
<br/>

If a relevant sentence falls at the boundary between two chunks, splitting without overlap would cut it in half — losing the context on either side. By repeating the last 50 characters of each chunk at the start of the next, every sentence is guaranteed to appear complete in at least one chunk.

Chunk size is a tradeoff: smaller chunks are more precise but lose surrounding context; larger chunks carry more context but reduce retrieval precision.

</details>

<details>
<summary>Why not just send the whole document to the LLM?</summary>
<br/>

LLMs have a context window — a hard limit on how much text they can process at once. `llama3.2:3b` supports 4096 tokens in this configuration, which is roughly 3000 words. Most real documents are longer than that.

Even when documents fit, sending everything degrades answer quality — the model gets confused by irrelevant content. RAG solves both problems: it keeps the context small and focused.

</details>

---

## Resources

- [Ollama documentation](https://ollama.com/docs)
- [ChromaDB documentation](https://docs.trychroma.com)
- [LangChain documentation](https://python.langchain.com)
- [sentence-transformers](https://www.sbert.net)
- [Llama 3.2 model card](https://ollama.com/library/llama3.2)

---

<div align="center">

Built by **Montassir Bouifraden** — Software engineering student, 1337 UM6P (42 Network)


</div>