
# Intelligent PDF Question Answering (RAG)

A Streamlit application that lets users upload a PDF and ask questions about its contents using Retrieval-Augmented Generation (RAG).

## Features

- Upload PDF documents through a web interface
- Extract text from PDF pages
- Split text into smaller chunks
- Generate semantic embeddings using Sentence Transformers
- Retrieve relevant passages using FAISS similarity search
- Generate answers using a local Llama 3.2 model through Ollama
- Display retrieved passages and PDF page references

## Tech Stack

- Python
- Streamlit
- LangChain
- Sentence Transformers
- FAISS
- Ollama (Llama 3.2)

## How It Works

1. Upload a PDF.
2. Extract and split its text into chunks.
3. Generate embeddings for the chunks.
4. Store the embeddings in a FAISS vector index.
5. Retrieve relevant passages for a user's question.
6. Pass the passages to the language model to generate a grounded answer.

## Prerequisites

- Python
- Git
- Ollama
- The Ollama model `llama3.2:3b`

## Setup

Clone the repository and enter the project folder.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the language model:

```bash
ollama pull llama3.2:3b
```

Start the application:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually
`http://localhost:8501`.

## Limitations

- The application works best with PDFs containing extractable text.
- Scanned PDFs may require OCR.
- Answer quality depends on the retrieved passages and language model.
- The local language model must be installed and available through Ollama.
