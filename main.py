from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq
from dotenv import load_dotenv
import json
import uvicorn
import io
import os

# Load environment variables from local .env file
load_dotenv()

app = FastAPI()

# Global In-Memory State
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_DIM = 384
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
chunk_store = []
indexed_files = set()
bm25_index = None
tokenized_corpus = []

def get_working_model_response(prompt_text, api_key=None, stream=True):
    key_to_use = api_key or GROQ_API_KEY
    if not key_to_use:
        raise ValueError("GROQ_API_KEY environment variable is not configured.")
        
    client = Groq(api_key=key_to_use)
    
    # Priority list containing ONLY strictly active, production Groq models
    candidate_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant"
    ]
    
    # Dynamically fetch available models from Groq account if possible
    try:
        remote_models = client.models.list()
        active_ids = [m.id for m in remote_models.data]
        # Filter candidate models to those actually active on your account
        valid_candidates = [m for m in candidate_models if m in active_ids]
        if valid_candidates:
            candidate_models = valid_candidates
        elif active_ids:
            candidate_models = active_ids
    except Exception as e:
        print(f"Could not list models dynamically: {e}. Falling back to default candidates.")

    last_error = None
    for model_name in candidate_models:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt_text}
                ],
                stream=stream
            )
            return client, response
        except Exception as e:
            print(f"Model {model_name} failed: {e}. Trying next...")
            last_error = e
            
    raise Exception(f"All Groq candidate models failed. Last Error: {last_error}")

def get_overlapping_chunks(text, filename, page_num, chunk_size=800, overlap=100):
    """Chunks text into overlapping segments, tagging each with metadata."""
    chunks = []
    if not text:
        return chunks
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append({
            "text": text[start:end].strip(),
            "filename": filename,
            "page": page_num
        })
        start += (chunk_size - overlap)
    return chunks

@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    global faiss_index, chunk_store, indexed_files, bm25_index, tokenized_corpus
    new_chunks = []
    
    for file in files:
        content = await file.read()
        filename = file.filename
        indexed_files.add(filename)
        
        if filename.lower().endswith('.pdf'):
            try:
                doc = fitz.open(stream=content, filetype="pdf")
                for page_num in range(len(doc)):
                    page_text = doc[page_num].get_text()
                    new_chunks.extend(get_overlapping_chunks(page_text, filename, page_num + 1))
            except Exception as e:
                print(f"Error parsing PDF {filename}: {e}")
        elif filename.lower().endswith('.txt'):
            try:
                text = content.decode('utf-8')
                new_chunks.extend(get_overlapping_chunks(text, filename, 1))
            except Exception as e:
                print(f"Error parsing TXT {filename}: {e}")
                
    if new_chunks:
        texts = [c["text"] for c in new_chunks]
        embeddings = embedding_model.encode(texts)
        faiss.normalize_L2(embeddings)
        faiss_index.add(np.array(embeddings).astype('float32'))
        chunk_store.extend(new_chunks)
        
        tokenized_corpus = [c["text"].lower().split() for c in chunk_store]
        bm25_index = BM25Okapi(tokenized_corpus)
        
    return {
        "status": "success",
        "total_chunks": len(chunk_store),
        "indexed_files": list(indexed_files)
    }

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query", "")
    api_key = data.get("api_key", "").strip() or GROQ_API_KEY
    history = data.get("history", [])
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")
    if not api_key:
        raise HTTPException(status_code=400, detail="Groq API Key is not configured in backend.")
        
    chitchat_keywords = {"hi", "hello", "hey", "thanks", "thank you", "good morning", "bye", "who are you"}
    normalized_query = query.strip().lower()
    is_chitchat = normalized_query in chitchat_keywords or (any(normalized_query.startswith(kw) for kw in ["hi", "hello", "hey", "who are you"]) and len(normalized_query) < 20)
    
    if is_chitchat:
        def generate_chitchat():
            yield "Hello! I'm ready to help you explore your uploaded documents. What would you like to learn or search for?"
            yield "\n__SOURCES__\n[]"
        return StreamingResponse(generate_chitchat(), media_type="text/plain")
        
    if faiss_index.ntotal == 0:
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Upload documents first.")
        
    try:
        typo_prompt = f"Fix any typos, spelling errors, or awkward phrasing in this user search query. Output ONLY the corrected, clean search query string and nothing else. Query: {query}"
        _, clean_query_response = get_working_model_response(typo_prompt, api_key, stream=False)
        cleaned_query = clean_query_response.choices[0].message.content.strip()
        if not cleaned_query:
            cleaned_query = query
    except Exception as e:
        print(f"Typo correction failed: {e}")
        cleaned_query = query
        
    summary_keywords = ["short notes", "summary", "summarize", "overview", "key points", "brief notes"]
    is_summary_query = any(kw in cleaned_query.lower() for kw in summary_keywords)
    
    explain_keywords = ["explain", "how does", "what is the mechanism", "teach me", "elaborate", "break down"]
    is_explain_query = any(kw in cleaned_query.lower() for kw in explain_keywords)
    
    search_query = cleaned_query
    if is_explain_query and history:
        last_msgs = [msg["content"] for msg in history[-2:] if "content" in msg]
        search_query = " ".join(last_msgs) + " " + cleaned_query
    
    retrieved_chunks = []
    if is_summary_query:
        retrieved_chunks = chunk_store[:8]
    else:
        query_emb = embedding_model.encode([search_query])
        faiss.normalize_L2(query_emb)
        
        fetch_k = min(20, faiss_index.ntotal)
        _, faiss_indices = faiss_index.search(np.array(query_emb).astype("float32"), fetch_k)
        faiss_ranks = {idx: rank for rank, idx in enumerate(faiss_indices[0]) if idx != -1 and idx < len(chunk_store)}
        
        bm25_ranks = {}
        if bm25_index is not None:
            tokenized_query = search_query.lower().split()
            bm25_scores = bm25_index.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_k]
            bm25_ranks = {idx: rank for rank, idx in enumerate(top_bm25_indices) if bm25_scores[idx] > 0 and idx < len(chunk_store)}
            
        rrf_scores = {}
        k = 60
        all_indices = set(faiss_ranks.keys()).union(set(bm25_ranks.keys()))
        for idx in all_indices:
            faiss_score = 1.0 / (k + faiss_ranks[idx]) if idx in faiss_ranks else 0.0
            bm25_score = 1.0 / (k + bm25_ranks[idx]) if idx in bm25_ranks else 0.0
            rrf_scores[idx] = faiss_score + bm25_score
            
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:7]
        chunk_indices = set(sorted_indices)
        retrieved_chunks = [chunk_store[idx] for idx in sorted_indices]
                
        if is_explain_query:
            tokens = [t.lower() for t in cleaned_query.split() if t.lower() not in explain_keywords and len(t) > 3]
            for i, chunk in enumerate(chunk_store):
                if i not in chunk_indices:
                    chunk_text_lower = chunk["text"].lower()
                    if any(tok in chunk_text_lower for tok in tokens):
                        retrieved_chunks.append(chunk)
                        chunk_indices.add(i)
                        if len(retrieved_chunks) >= 12:
                            break
                            
    if retrieved_chunks:
        query_emb_for_scoring = embedding_model.encode([cleaned_query])
        faiss.normalize_L2(query_emb_for_scoring)
        query_emb_np = np.array(query_emb_for_scoring).astype("float32").flatten()
        
        chunk_texts = [c["text"] for c in retrieved_chunks]
        chunk_embs = embedding_model.encode(chunk_texts)
        faiss.normalize_L2(chunk_embs)
        
        scored_chunks = []
        for c, chunk_vec in zip(retrieved_chunks, chunk_embs):
            c_copy = c.copy()
            cos_sim = np.dot(query_emb_np, np.array(chunk_vec).astype("float32"))
            percent_score = max(0.0, min(100.0, float(cos_sim) * 100))
            c_copy["score"] = round(percent_score, 1)
            scored_chunks.append(c_copy)
        retrieved_chunks = scored_chunks
    
    context_str = "\n\n---\n\n".join([
        f"Source: {c['filename']} (Page {c['page']})\n{c['text']}" for c in retrieved_chunks
    ])
    
    prompt_query_section = f"USER QUESTION (Cleaned):\n{cleaned_query}\n\n(Original User Query: {query})"
    
    if is_summary_query:
        prompt = f"""You are an academic research assistant. Provide a clear, structured summary with key bullet points strictly based on these excerpts:

Context:
{context_str}

{prompt_query_section}"""
    elif is_explain_query:
        prompt = f"""You are an expert academic tutor. Synthesize the context into a structured explanation.

EXPLANATION STRUCTURE:
1. 🎯 **The Core Concept**: High-level intuition in 1-2 simple sentences.
2. ⚙️ **How It Works**: Step-by-step breakdown using bullet points.
3. 💡 **Key Terms & Context**: Simple definitions of technical terms.
4. 📌 **Why It Matters**: Primary importance based on the text.

CONTEXT SNIPPETS:
{context_str}

{prompt_query_section}"""
    else:
        prompt = f"""You are a helpful knowledge assistant. Answer strictly using the context below. If the answer cannot be found, state "I cannot find the answer in the provided documents".

Context:
{context_str}

{prompt_query_section}"""

    def generate_response():
        try:
            client, response = get_working_model_response(prompt, api_key, stream=True)
            
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
                    
            try:
                sug_prompt = f"""Generate EXACTLY 3 short follow-up questions that the user can ask next. Output ONLY a valid JSON array of 3 strings.

CONTEXT EXCERPTS:
{context_str}"""
                _, sug_resp = get_working_model_response(sug_prompt, api_key, stream=False)
                sug_text = sug_resp.choices[0].message.content.strip().removeprefix("```json").removesuffix("```").strip()
                yield f"\n__SUGGESTIONS__\n{sug_text}"
            except Exception:
                pass
            
            yield "\n__SOURCES__\n"
            yield json.dumps(retrieved_chunks)
        except Exception as e:
            error_msg = f"**System Alert:** The AI models encountered an error and could not generate a response.\n\n`Details: {str(e)}`"
            yield error_msg
            yield "\n__SOURCES__\n[]"
            
    return StreamingResponse(generate_response(), media_type="text/plain")

@app.post("/api/clear")
async def clear_memory():
    global faiss_index, chunk_store, indexed_files, bm25_index, tokenized_corpus
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    chunk_store = []
    indexed_files = set()
    bm25_index = None
    tokenized_corpus = []
    return {"status": "cleared"}

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)