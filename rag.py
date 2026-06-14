# rag.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ─── DOCUMENT ────────────────────────────────────────────────────────────────
# This is our knowledge base. In a real system this would be loaded from
# a PDF or database. We hardcode it here to keep things simple.
# The content is about AI systems and backend architecture — directly
# relevant to AI Movement UM6P.

DOCUMENT = """
# AI Systems and Backend Architecture

## What is a Large Language Model
A Large Language Model (LLM) is a neural network trained on massive amounts 
of text data. During training, the model learns statistical patterns between 
words, sentences, and concepts. It does not memorize text — it learns 
relationships. Models like Llama, GPT, and Claude contain billions of 
parameters that encode these learned patterns.

LLMs have a context window — a maximum amount of text they can process 
at once. GPT-4 has a 128,000 token context window. Llama 3.2 3B has 
a 128,000 token context window as well. One token is approximately 
0.75 words in English, and 0.5 words in Arabic.

## What are Embeddings
An embedding is a numerical representation of text as a vector — a list 
of floating point numbers. The key property is that semantically similar 
texts produce similar vectors. This allows mathematical operations on meaning.

For example, using a 384-dimensional embedding model:
- "Payment is due in 30 days" → [0.82, 0.31, 0.76, ...]
- "Invoice must be settled within a month" → [0.79, 0.33, 0.71, ...]
- "Docker runs isolated containers" → [0.11, 0.89, 0.23, ...]

The first two sentences are semantically close — their vectors are similar.
The third is semantically different — its vector is far from the others.

## What is RAG
Retrieval Augmented Generation (RAG) is a technique that enhances LLM 
responses by providing relevant context retrieved from a knowledge base.

The RAG pipeline has two phases:

Indexing phase (done once):
1. Split documents into chunks of approximately 500 words
2. Convert each chunk to an embedding vector
3. Store chunks and vectors in a vector database

Query phase (every question):
1. Convert the question to an embedding vector
2. Find the most similar chunk vectors in the database
3. Build a prompt combining the question and retrieved chunks
4. Send the prompt to the LLM
5. Return the generated answer

RAG solves the context window limitation and grounds LLM responses 
in specific, up-to-date documents rather than training data alone.

## Vector Databases
A vector database stores embeddings and enables similarity search.
Unlike traditional databases that match exact values, vector databases 
find the nearest neighbors in high-dimensional space.

ChromaDB is an open-source vector database that runs locally in memory 
or persisted to disk. It is suitable for development and small-scale 
production systems. For large-scale production, alternatives include 
Pinecone, Weaviate, and pgvector (PostgreSQL extension).

## Backend Infrastructure for AI Systems
AI systems require robust backend infrastructure:

API Layer: REST or GraphQL APIs that expose model inference endpoints.
These handle authentication, rate limiting, request validation, and 
response formatting. Express.js and FastAPI are common choices.

Async Processing: LLM inference is slow — typically 1-30 seconds per 
request. Production systems use job queues (BullMQ, Celery) to handle 
requests asynchronously, preventing API timeouts and enabling retries.

Caching: Identical or similar queries can be cached. Redis is commonly 
used to cache embeddings and model responses, reducing latency and cost.

Containerization: AI models and their dependencies are packaged in 
Docker containers for reproducible deployment. GPU-enabled containers 
require the NVIDIA Container Toolkit.

Observability: Prometheus metrics track inference latency, token usage, 
queue depth, and error rates. Grafana dashboards visualize these metrics.

## Arabic NLP Challenges
Processing Arabic text presents specific challenges:

Morphological complexity: Arabic words carry grammatical information 
through prefixes, suffixes, and internal vowel changes. One root can 
generate hundreds of word forms.

Right-to-left rendering: Text direction affects UI and preprocessing.

Dialectal variation: Modern Standard Arabic differs significantly from 
Moroccan Darija and other dialects. Models trained on MSA perform poorly 
on dialectal text.

Tokenization: Arabic tokenizers must handle connected script and 
diacritics correctly. Incorrect tokenization degrades embedding quality.

Current research at AI Movement UM6P focuses on developing Arabic-specific 
embedding models and improving LLM performance on Moroccan dialect text.
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # runs locally, no API key
LLM_MODEL = "llama3.2:3b"             # the model we pulled with ollama
CHUNK_SIZE = 500                        # words per chunk approximately
CHUNK_OVERLAP = 50                      # overlap between chunks
N_RESULTS = 3                           # how many chunks to retrieve per question

# ─── INDEXING PIPELINE ───────────────────────────────────────────────────────
def build_vectorstore():
    print("\n📄 Splitting document into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.create_documents([DOCUMENT])
    print(f"   → {len(chunks)} chunks created")

    print("\n🧠 Loading embedding model (downloads on first run ~80MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    print("   → Embedding model ready")

    print("\n📦 Storing chunks in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    print(f"   → {len(chunks)} chunks embedded and stored")

    return vectorstore

# ─── PROMPT TEMPLATE ─────────────────────────────────────────────────────────
def build_prompt():
    template = """You are an AI assistant with expertise in AI systems,
backend architecture, and data science. You answer questions clearly
and concisely based on the provided context.

If the answer is not in the context, say honestly:
"I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )

# ─── RETRIEVAL CHAIN ─────────────────────────────────────────────────────────
def build_chain(vectorstore, prompt):
    llm = Ollama(
        model=LLM_MODEL,
        temperature=0.1,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": N_RESULTS}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return chain

# ─── MAIN LOOP ───────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("   Arabic/French Document Q&A System")
    print("   Powered by Llama3.2 + ChromaDB + sentence-transformers")
    print("="*60)

    print("\n⚙️  Building knowledge base...")
    vectorstore = build_vectorstore()

    print("\n⚙️  Setting up retrieval chain...")
    prompt = build_prompt()
    chain = build_chain(vectorstore, prompt)

    print("\n✅ System ready. Type your question below.")
    print("   Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() == "quit":
            print("\nGoodbye.")
            break

        print("\n🔍 Searching knowledge base...")
        result = chain.invoke({"query": question})

        print("\n🤖 Answer:")
        print(result["result"])

        print("\n📚 Sources used:")
        for i, doc in enumerate(result["source_documents"], 1):
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"   [{i}] {preview}...")

        print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    main()
