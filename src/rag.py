from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

embedding_model = SentenceTransformer("all-miniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chunk_text(text, chunk_size=200, overlap=30):
    """
    Splits long text into smaller overlapping chunks
    Why? Because FAISS works better with smaller focused pieces
    overlap=50 means chunks share 50 chars — prevents cutting mid-sentence
    
    Example:
    text = "ABCDEFGHIJ"
    chunk_size=4, overlap=2
    chunks = ["ABCD", "CDEF", "EFGH", "GHIJ"]
    """

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_faiss_index(credible_results):
    """
    Takes credible results → chunks text → embeds → stores in FAISS index
    
    FAISS = Facebook AI Similarity Search
    It stores vectors and lets us find most similar ones instantly
    Think of it as a super fast 'nearest neighbor' search
    """
    all_chunks = []
    all_metadata = []
    credible_results = credible_results[:5]

    for r in credible_results:
        text = r["full_text"]
        if not text or len(text)<50:
            continue

        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                "title": r["title"],
                "url": r["url"],
                "credibility_score": r["credibility_score"],
                "credibility_label": r["credibility_label"]
            })
    
    if not all_chunks:
        return None, [], []
    
    # Convert chunks to embeddings — each chunk becomes a 384-dim vector
    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embedding_model.encode(all_chunks, convert_to_numpy=True)

    # Build FAISS index
    # IndexFlatL2 = finds vectors with smallest L2 (euclidean) distance
    dimension = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))  # FAISS needs float32

    print(f"FAISS index built with {index.ntotal} vectors!")
    return index, all_chunks, all_metadata

def retrieve_relevant_chunks(query, index, all_chunks, all_metadata, top_k=3):
    """
    Given a user query — find the most relevant chunks from FAISS
    
    How it works:
    1. Convert query to embedding vector
    2. FAISS finds top_k closest vectors
    3. Return those chunks + their source info
    """
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    
    # Search FAISS — returns distances and indices of top_k matches
    distances, indices = index.search(query_embedding.astype(np.float32), top_k)
    
    relevant = []
    for i, idx in enumerate(indices[0]):
        if idx < len(all_chunks):
            relevant.append({
                "chunk": all_chunks[idx],
                "metadata": all_metadata[idx],
                "distance": float(distances[0][i])
            })
    return relevant

def generate_summary(condition, relevant_chunks):
    """
    Takes retrieved chunks → sends to Groq → gets plain English summary
    
    This is the RAG part:
    R = Retrieved chunks from FAISS (Retrieval)
    A = Passed as context to LLM (Augmented)
    G = LLM generates answer grounded in context (Generation)
    """
    # Build context from retrieved chunks
    context = ""
    sources = set()
    for item in relevant_chunks:
        context += f"\n---\n{item['chunk']}"
        sources.add(item['metadata']['url'])

    # Prompt engineering — tell LLM exactly what to do
    prompt = f"""You are a helpful medical information assistant.
Based ONLY on the following verified medical sources, provide a clear and simple summary about {condition}.

CONTEXT FROM VERIFIED SOURCES:
{context}

Your response must:
1. Explain what {condition} is in simple language
2. List main symptoms
3. List evidence-based treatments
4. Mention what is still debated or unclear
5. End with: "Please consult a qualified doctor for personal medical advice."

Do NOT add any information not present in the context above.
Keep response under 300 words.
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",  # free, fast Groq model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3  # low temperature = more factual, less creative
    )

    summary = response.choices[0].message.content

    return {
        "summary": summary,
        "sources": list(sources)
    }

def run_rag_pipeline(condition, credible_results):
    """
    Master function — ties everything together
    Input: condition + credible results from scraper
    Output: plain English summary + sources
    """
    # Step 1 — Build FAISS index from credible results
    credible_results = credible_results[:5]
    index, all_chunks, all_metadata = build_faiss_index(credible_results)

    if index is None:
        return {"summary": "No credible content found.", "sources": []}

    # Step 2 — Retrieve most relevant chunks for the condition
    relevant_chunks = retrieve_relevant_chunks(condition, index, all_chunks, all_metadata, top_k=3)

    # Step 3 — Generate summary using Groq
    result = generate_summary(condition, relevant_chunks)

    return result