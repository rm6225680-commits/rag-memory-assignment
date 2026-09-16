import os
import glob
import hashlib
from dotenv import load_dotenv as load
import chromadb
from google import genai

# Load environment variables from .env file
load()

# Initialize the Gemini client using the API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing! Please set it in your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)
EMBEDDING_MODEL = "text-embedding-004"

# Initialize Persistent ChromaDB Client
# This saves your vector database locally to disk so data persists across restarts
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_memories")


def chunk_text(text, chunk_size=300, overlap=50):
    """Splits raw text into smaller chunks with character overlap to preserve context."""
    chunks = []
    start = 0
    text_length = len(text)
    
    if text_length == 0:
        return chunks

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_embedding(text: str):
    """Generates a vector embedding for a given text using Gemini."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )
    # Extract embedding values from the Gemini SDK response
    return response.embeddings[0].values


def ingest_documents(data_dir="data"):
    """Reads all text documents from the data directory, chunks them, embeds them, and stores them in ChromaDB."""
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    if not txt_files:
        print(f"Warning: No text files found in '{data_dir}' directory.")
        return 0

    total_added = 0

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        print(f"Processing file: {filename}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        chunks = chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            # Create a unique, stable ID for the chunk using hashing to prevent duplicate ingestion
            chunk_id = hashlib.md5(f"{filename}_{i}_{chunk}".encode()).hexdigest()
            
            # Check if this chunk already exists in ChromaDB to avoid redundant embedding generation
            existing = collection.get(ids=[chunk_id])
            if existing and existing["ids"]:
                continue  # Already stored, skip

            # Generate vector embedding via Gemini API
            vector = get_embedding(chunk)
            
            # Store in ChromaDB
            collection.add(
                ids=[chunk_id],
                embeddings=[vector],
                documents=[chunk],
                metadatas=[{"source": filename, "chunk_index": i}]
            )
            total_added += 1

    print(f"Successfully ingested {total_added} new memory chunks into ChromaDB.")
    return total_added


def retrieve_memories(query: str, n_results: int = 2):
    """Embeds a query and retrieves the most similar text chunks from ChromaDB."""
    if collection.count() == 0:
        print("Database is empty. Please ingest documents first.")
        return {"documents": [[]], "metadatas": [[]]}

    query_vector = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(n_results, collection.count())
    )
    return results