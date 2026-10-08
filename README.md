# ⚡ NEXUS AI — Intelligent RAG Knowledge Assistant

NEXUS AI is an enterprise-grade Retrieval-Augmented Generation (RAG) document assistant designed to eliminate traditional keyword search (`Ctrl+F`) limitations. It converts uploaded PDFs and text notes into 384-dimensional dense vector representations, enabling high-precision semantic retrieval with exact source page citations and pedagogical AI explanations.

---

## ✨ Key Features

* **Multi-Format Ingestion:** Drag-and-drop support for multi-page `.pdf` and `.txt` documents using PyMuPDF parsing.
* **Dense Vector Search:** Fast cosine similarity indexing using FAISS (`IndexFlatIP`) and `sentence-transformers` (`all-MiniLM-L6-v2`).
* **Pedagogical AI Synthesis:** System prompts structure responses into **Core Concepts**, **Step-by-Step Mechanisms**, **Jargon Translation**, and **Key Takeaways**.
* **Interactive Source Citations:** Expandable citation cards detailing exact page numbers and retrieved text snippets.
* **API Resilience Loop:** Dynamic multi-model fallback (`gemini-2.0-flash` → `gemini-1.5-flash` → `gemini-2.5-flash`) handling 429 quota and 404 availability errors automatically.
* **Cyber-Dark SaaS UI:** Responsive glassmorphic interface powered by Tailwind CSS and Lucide icons.

---

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI (Python)
* **Vector Engine:** FAISS (`faiss-cpu`)
* **Embeddings Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **LLM Engine:** Google Gemini API (`2.0-flash` / `1.5-flash`)
* **PDF Engine:** PyMuPDF (`fitz`)
* **Frontend:** Single-page HTML5, Tailwind CSS, JavaScript Fetch API

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/cramkarthik123-lab/Nexusclub-.git
cd Nexusclub-
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Backend Server
```bash
uvicorn main:app --reload
```

### 4. Open in Browser
Navigate to `[http://127.0.0.1:8000](http://127.0.0.1:8000)` to launch the NEXUS AI workspace.
