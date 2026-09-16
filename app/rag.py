import os
from dotenv import load_dotenv
from google import genai
from app.memory_store import retrieve_memories, ingest_documents, collection

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GENERATION_MODEL = "gemini-2.5-flash"


def answer_query(query: str):
    """Answers a user query using RAG: retrieves relevant memories, displays them, and generates an LLM response."""
    print(f"\n--- User Query: '{query}' ---")
    
    # 1. Retrieve relevant memories from ChromaDB
    results = retrieve_memories(query, n_results=2)
    
    # Extract documents and metadatas safely
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    
    # 2. Inspect / Display retrieved context (Recruiter requirement)
    print("\n[Inspecting Retrieved Memories/Context]:")
    if not documents:
        print("-> No relevant memories found in the database.")
        context_text = "No relevant context available."
    else:
        for idx, (doc, meta) in enumerate(zip(documents, metadatas)):
            source = meta.get("source", "unknown") if meta else "unknown"
            print(f"  [{idx+1}] Source: {source} | Content: {doc}")
        context_text = "\n\n".join(documents)

    # 3. Construct prompt for Gemini
    prompt = f"""
You are a helpful AI assistant with access to a persistent memory store.
Answer the user's query based ONLY on the provided context below. If the context does not contain enough information to answer the query, clearly state that you do not have enough information in your memories to answer. Do not make up facts.

Context:
{context_text}

User Query:
{query}
"""

    # 4. Generate answer using Gemini
    print("\n[Generating Answer from Gemini]:")
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )
    
    answer = response.text.strip()
    print(f"\nAnswer:\n{answer}\n")
    return answer