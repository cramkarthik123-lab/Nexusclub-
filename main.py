from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import google.generativeai as genai
import json
import uvicorn
import io
import os

app = FastAPI()

# Global In-Memory State
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6IONIo2rITrMhUIuyqbzCmkAsngnl_P6S8dgN4EH3RWsA")
EMBEDDING_DIM = 384
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
chunk_store = []
indexed_files = set()
bm25_index = None
tokenized_corpus = []

def get_working_model_response(prompt_text, api_key, stream=True):
    genai.configure(api_key=api_key)
    try:
        available_models = [m.name.replace("models/", "") for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    except Exception as e:
        print(f"Error fetching models: {e}")
        available_models = []
        
    PRIORITY_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro-latest"
    ]
    
    candidates = [m for m in PRIORITY_MODELS if m in available_models] if available_models else PRIORITY_MODELS
    if not candidates:
        candidates = PRIORITY_MODELS
        
    last_error = None
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            if stream:
                response = model.generate_content(prompt_text, stream=True)
                iterator = iter(response)
                first_chunk = next(iterator, None)
                print(f"Successfully connected to model (stream): {model_name}")
                return model, iterator, first_chunk
            else:
                response = model.generate_content(prompt_text)
                print(f"Successfully connected to model: {model_name}")
                return model, response
        except Exception as e:
            print(f"Model {model_name} failed: {e}. Trying next...")
            last_error = e
            
    raise Exception(f"All candidate models failed. Last Error: {last_error}")

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
                # Parse PDF page-by-page using PyMuPDF (fitz)
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
        # Generate and normalize embeddings for cosine similarity (IndexFlatIP)
        texts = [c["text"] for c in new_chunks]
        embeddings = embedding_model.encode(texts)
        faiss.normalize_L2(embeddings)
        faiss_index.add(np.array(embeddings).astype('float32'))
        chunk_store.extend(new_chunks)
        
        # Rebuild BM25 index
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
    api_key = data.get("api_key", "").strip() or DEFAULT_GEMINI_API_KEY
    history = data.get("history", [])
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key is not configured in backend.")
        
    # Conversational & Chitchat Router
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
        
    # Query Typo Correction & Normalization
    try:
        typo_prompt = f"Fix any typos, spelling errors, or awkward phrasing in this user search query. Output ONLY the corrected, clean search query string and nothing else. Query: {query}"
        _, clean_query_response = get_working_model_response(typo_prompt, api_key, stream=False)
        cleaned_query = clean_query_response.text.strip()
        if not cleaned_query:
            cleaned_query = query
    except Exception as e:
        print(f"Typo correction failed: {e}")
        cleaned_query = query
        
    # Intent Detection
    summary_keywords = ["short notes", "summary", "summarize", "overview", "key points", "brief notes"]
    is_summary_query = any(kw in cleaned_query.lower() for kw in summary_keywords)
    
    explain_keywords = ["explain", "how does", "what is the mechanism", "teach me", "elaborate", "break down"]
    is_explain_query = any(kw in cleaned_query.lower() for kw in explain_keywords)
    
    # Query refinement for follow-ups
    search_query = cleaned_query
    if is_explain_query and history:
        last_msgs = [msg["content"] for msg in history[-2:] if "content" in msg]
        search_query = " ".join(last_msgs) + " " + cleaned_query
    
    retrieved_chunks = []
    if is_summary_query:
        # Retrieve the first 8 chunks from memory for broad summary context
        retrieved_chunks = chunk_store[:8]
    else:
        # Hybrid Search: FAISS + BM25 with RRF
        query_emb = embedding_model.encode([search_query])
        faiss.normalize_L2(query_emb)
        
        # 1. FAISS Search (top 20)
        fetch_k = min(20, faiss_index.ntotal)
        _, faiss_indices = faiss_index.search(np.array(query_emb).astype("float32"), fetch_k)
        faiss_ranks = {idx: rank for rank, idx in enumerate(faiss_indices[0]) if idx != -1 and idx < len(chunk_store)}
        
        # 2. BM25 Search (top 20)
        bm25_ranks = {}
        if bm25_index is not None:
            tokenized_query = search_query.lower().split()
            bm25_scores = bm25_index.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_k]
            bm25_ranks = {idx: rank for rank, idx in enumerate(top_bm25_indices) if bm25_scores[idx] > 0 and idx < len(chunk_store)}
            
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k = 60
        all_indices = set(faiss_ranks.keys()).union(set(bm25_ranks.keys()))
        for idx in all_indices:
            faiss_score = 1.0 / (k + faiss_ranks[idx]) if idx in faiss_ranks else 0.0
            bm25_score = 1.0 / (k + bm25_ranks[idx]) if idx in bm25_ranks else 0.0
            rrf_scores[idx] = faiss_score + bm25_score
            
        # Sort by RRF score and get top 7
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:7]
        
        chunk_indices = set(sorted_indices)
        retrieved_chunks = [chunk_store[idx] for idx in sorted_indices]
                
        # Hybrid Keyword Fallback Search
        if is_explain_query:
            tokens = [t.lower() for t in cleaned_query.split() if t.lower() not in explain_keywords and len(t) > 3]
            for i, chunk in enumerate(chunk_store):
                if i not in chunk_indices:
                    chunk_text_lower = chunk["text"].lower()
                    if any(tok in chunk_text_lower for tok in tokens):
                        retrieved_chunks.append(chunk)
                        chunk_indices.add(i)
                        if len(retrieved_chunks) >= 12: # Cap total chunks
                            break
                            
    if retrieved_chunks:
        # Calculate Cosine Similarity Percentages
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
    
    # Construct Context
    context_str = "\n\n---\n\n".join([
        f"Source: {c['filename']} (Page {c['page']})\n{c['text']}" for c in retrieved_chunks
    ])
    
    # Unified Query Prompting
    prompt_query_section = f"USER QUESTION (Cleaned):\n{cleaned_query}\n\n(Original User Query: {query})"
    
    if is_summary_query:
        # Prompt Adjustment for broad queries
        prompt = f"""You are an academic research assistant. The user is requesting short notes/a summary based on the provided document context below. Provide a clear, structured summary with key bullet points based strictly on these excerpts:

Context:
{context_str}

{prompt_query_section}"""
    elif is_explain_query:
        # Pedagogical Explanation Prompt
        prompt = f"""You are an expert academic tutor and conceptual teacher.
The user wants an intuitive, thorough explanation of a concept based on the document excerpts provided below.

YOUR GOAL:
Do NOT just restate or copy sentences from the context verbatim. Instead, synthesize the provided facts into a structured, easy-to-understand explanation as if you were teaching a student.

EXPLANATION STRUCTURE TO FOLLOW:
1. 🎯 **The Core Concept (High-Level Intuition)**: Explain what this topic is in 1-2 simple sentences using plain English.
2. ⚙️ **How It Works (Step-by-Step Breakdown)**: Walk through the mechanism or key points logically using bullet points.
3. 💡 **Key Terms & Context**: Translate any difficult jargon or technical terms mentioned in the text into simple definitions.
4. 📌 **Why It Matters / Takeaway**: Summarize the primary importance of this concept based on the text.

STRICT GROUNDING RULES:
- Use ONLY facts, definitions, and relationships stated in the provided context snippets.
- Do not invent external facts or outside theories not present in the document.
- If the context mentions the topic but lacks full detail, explain what IS available in the context and note what details are missing.
- If the topic is completely absent from the context, state: 'I cannot find information about this topic in the provided documents.'

CONTEXT SNIPPETS:
{context_str}

{prompt_query_section}"""
    else:
        # Grounding Prompt for factual queries
        prompt = f"""You are a helpful knowledge assistant. Please answer the user's question based strictly on the provided context below. Do not use outside knowledge. If the answer cannot be found in the provided context, state "I cannot find the answer in the provided documents".

Context:
{context_str}

{prompt_query_section}"""

    def generate_response():
        try:
            active_model, iterator, first_chunk = get_working_model_response(prompt, api_key, stream=True)
            
            if first_chunk and first_chunk.text:
                yield first_chunk.text
                
            for chunk in iterator:
                if chunk.text:
                    yield chunk.text
                    
            # Generate suggestions using a strictly grounded prompt
            try:
                sug_prompt = f"""Generate EXACTLY 3 short follow-up questions that the user can ask next. Output ONLY a valid JSON array of 3 strings.

CRITICAL REQUIREMENT: Every suggested question MUST be answerable strictly using the provided document context snippets below. Do NOT suggest questions about related general topics, broader concepts, or missing sections that are not explicitly detailed in the context snippets.

CONTEXT EXCERPTS:
{context_str}

Based ONLY on the facts present in the excerpts above, provide 3 clean follow-up questions that probe deeper into these specific facts."""
                sug_resp = active_model.generate_content(sug_prompt)
                sug_text = sug_resp.text.strip().removeprefix("```json").removesuffix("```").strip()
                yield f"\n__SUGGESTIONS__\n{sug_text}"
            except Exception as e:
                pass # skip suggestions if generation fails
            
            # Send citations metadata at the end separated by a special delimiter
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
