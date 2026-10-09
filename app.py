import streamlit as st
import fitz  # PyMuPDF
import io
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

def inject_custom_css():
    """Injects custom CSS for a modern dark glassmorphism UI."""
    st.markdown("""
    <style>
    /* Modern dark background */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    
    /* Glassmorphism styling for sidebar and inputs */
    [data-testid="stSidebar"] {
        background: rgba(15, 32, 39, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Custom styled source cards */
    .source-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-left: 4px solid #4facfe;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .source-header {
        color: #4facfe;
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    /* Dynamic metric counters */
    .metric-container {
        display: flex;
        justify-content: space-between;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-top: 1rem;
    }
    .metric-box h3 {
        margin: 0;
        font-size: 1.8rem;
        color: #4facfe;
    }
    .metric-box p {
        margin: 0;
        font-size: 0.85rem;
        color: #b0bec5;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    """, unsafe_allow_html=True)

def extract_and_chunk_text(file_obj, filename, chunk_size=800, overlap=100):
    """
    Extracts text from a file object (PDF via PyMuPDF or TXT) and breaks it into overlapping chunks.
    Returns a list of dictionaries containing the chunk text and the source filename.
    """
    text = ""
    file_type = filename.split('.')[-1].lower()
    
    # Reset file pointer
    file_obj.seek(0)
    
    # Extract text based on file type
    if file_type == "pdf":
        try:
            # Use PyMuPDF (fitz) for PDF parsing
            doc = fitz.open(stream=file_obj.read(), filetype="pdf")
            for page in doc:
                text += page.get_text() + "\n"
        except Exception as e:
            st.error(f"Error reading PDF {filename}: {e}")
            return []
    elif file_type == "txt":
        try:
            text = file_obj.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading TXT {filename}: {e}")
            return []
    else:
        st.error(f"Unsupported file type for {filename}.")
        return []

    # Chunk the text with overlap
    chunks = []
    if not text:
        return chunks

    start = 0
    text_length = len(text)

    # Create overlapping chunks
    while start < text_length:
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            "text": chunk_text,
            "filename": filename
        })
        start += (chunk_size - overlap)

    return chunks

@st.cache_resource
def load_embedding_model():
    """
    Loads and caches the SentenceTransformer model to prevent reloading on every interaction.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_faiss_index_and_search(chunk_dicts, query, top_k=4):
    """
    Creates a FAISS index from the text chunks using Cosine Similarity (IndexFlatIP with normalized vectors), 
    embeds the query, and performs a similarity search.
    Returns the top matching chunk dictionaries.
    """
    if not chunk_dicts or not query:
        return []
    
    model = load_embedding_model()
    
    # Extract just the text for embedding
    texts = [c["text"] for c in chunk_dicts]
    
    # Generate embeddings for the chunks
    chunk_embeddings = model.encode(texts)
    
    # Normalize embeddings for Cosine Similarity
    faiss.normalize_L2(chunk_embeddings)
    
    # Initialize FAISS index using Inner Product (Cosine Similarity for normalized vectors)
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to the index
    index.add(np.array(chunk_embeddings).astype("float32"))
    
    # Embed the search query and normalize
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Perform the search
    distances, indices = index.search(np.array(query_embedding).astype("float32"), min(top_k, len(chunk_dicts)))
    
    # Retrieve the top matching chunk dictionaries
    results = []
    for idx in indices[0]:
        if idx < len(chunk_dicts):
            results.append(chunk_dicts[idx])
            
    return results

def generate_gemini_response_stream(query, context_chunks, api_key):
    """
    Generates a streaming response from the Gemini model using the retrieved chunks as strict context.
    Yields text chunks for st.write_stream.
    """
    if not api_key:
        yield "Error: Gemini API Key is missing. Please provide it in the sidebar."
        return
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Combine the text from the context chunks
        context_texts = [c["text"] for c in context_chunks]
        context_str = "\n\n---\n\n".join(context_texts)
        
        # Construct the strict prompt
        prompt = f"""You are a helpful assistant. Please answer the user's question based strictly on the provided context below. Do not use outside knowledge. If the answer cannot be found in the context, state that you do not know.

Context:
{context_str}

User Question: {query}
"""
        
        # Stream the response
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error generating response: {e}"

def main():
    st.set_page_config(page_title="Glass RAG AI", layout="wide")
    inject_custom_css()
    
    # Initialize session state for chat messages if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar Configuration
    st.sidebar.title("✨ Glass RAG AI")
    st.sidebar.markdown("Modern RAG with PyMuPDF, Cosine Similarity & Streaming")
    
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Enter your API key here...")
    
    if not gemini_api_key:
        st.sidebar.warning("API Key required.")
    else:
        st.sidebar.success("API Key active.")
        
    st.sidebar.divider()
    
    # Sidebar File Upload
    st.sidebar.subheader("📄 Knowledge Base")
    uploaded_files = st.sidebar.file_uploader(
        "Upload .pdf and .txt files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    # Process uploaded files
    all_chunks = []
    if uploaded_files:
        with st.sidebar.status("Processing Documents...", expanded=True) as status:
            for uploaded_file in uploaded_files:
                # Extract and chunk with PyMuPDF
                chunks = extract_and_chunk_text(
                    file_obj=uploaded_file, 
                    filename=uploaded_file.name,
                    chunk_size=800, 
                    overlap=100
                )
                if chunks:
                    all_chunks.extend(chunks)
            status.update(label="Processing Complete!", state="complete", expanded=False)
                
        if all_chunks:
            # Dynamic Metric Counters
            st.sidebar.markdown(f"""
            <div class="metric-container">
                <div class="metric-box">
                    <h3>{len(uploaded_files)}</h3>
                    <p>Files</p>
                </div>
                <div class="metric-box">
                    <h3>{len(all_chunks)}</h3>
                    <p>Chunks</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Main Chat UI
    st.title("Chat with your Documents")
    
    # Display existing chat messages from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # If the message is from the assistant and has context, display the styled expandable sources
            if message["role"] == "assistant" and "context" in message:
                with st.expander("🔍 View Retrieved Sources"):
                    for i, source in enumerate(message["context"]):
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">Source {i+1}: {source['filename']}</div>
                            <div>{source['text']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
    # Chat input area
    if prompt := st.chat_input("Ask a question about your documents..."):
        # 1. Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 2. Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 3. Generate and display assistant response
        with st.chat_message("assistant"):
            if not all_chunks:
                response = "Please upload some documents in the sidebar first so I have context to answer your questions."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            elif not gemini_api_key:
                response = "Please enter your Gemini API Key in the sidebar to get an answer."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Retrieve relevant chunks using FAISS (IndexFlatIP Cosine Similarity)
                top_chunks = create_faiss_index_and_search(all_chunks, prompt, top_k=4)
                
                # Generate response with Gemini using token streaming
                stream = generate_gemini_response_stream(prompt, top_chunks, gemini_api_key)
                
                # Display the streamed text using st.write_stream
                full_response = st.write_stream(stream)
                
                # Display the sources in an expander using Custom CSS Glass Cards
                with st.expander("🔍 View Retrieved Sources"):
                    for i, source in enumerate(top_chunks):
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header">Source {i+1}: {source['filename']}</div>
                            <div>{source['text']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                            
                # Add assistant response and context to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "context": top_chunks
                })

if __name__ == "__main__":
    main()
