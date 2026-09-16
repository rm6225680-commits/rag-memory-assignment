# RAG Memory Assistant (Gemini API & ChromaDB)

A lightweight Retrieval-Augmented Generation (RAG) system built with Python, Google's `google-genai` SDK (`gemini-3.6-flash`), and ChromaDB for local persistent vector storage.

## Features
- **Document Ingestion & Chunking:** Automatically loads text files from a `data/` folder, splits them into manageable chunks, and creates vector embeddings.
- **Persistent Vector Database:** Uses ChromaDB to store text chunks and embeddings locally on disk.
- **Context Inspection:** Displays retrieved context chunks and their sources directly in the terminal before generating answers.
- **Grounded LLM Responses:** Uses Gemini to answer user questions strictly based on retrieved memory context, avoiding hallucinations.

## Project Structure
```text
rag-memory-assignment/
│
├── app/
│   ├── __init__.py
│   ├── memory_store.py   # Handles ChromaDB, chunking, and embeddings
│   ├── rag.py            # Handles context formatting and Gemini generation
│   └── main.py           # CLI entry point and interactive loop
│
├── data/
│   ├── conversation1.txt # Sample memory data
│   └── conversation2.txt # Sample memory data
│
├── .env                  # API keys (ignored by git)
├── .gitignore            # Ignores venv, chromadb, and .env
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation

Setup & Installation Instructions
If you want to clone and run this repository locally, follow these steps:

Clone the repository:

Bash
git clone [https://github.com/rm6225680-commits/rag-memory-assignment.git](https://github.com/rm6225680-commits/rag-memory-assignment.git)
cd rag-memory-assignment
Create and activate a virtual environment:

Bash
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
Install dependencies:

Bash
pip install -r requirements.txt
Configure your environment variables:
Create a .env file in the root directory and add your Google Gemini API key:

Code snippet
GEMINI_API_KEY=your_actual_api_key_here
Run the Application:

Bash
python -m app.main