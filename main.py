
import os
import io
import json
import tempfile
from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from sentence_transformers import SentenceTransformer
import numpy as np
import requests

app = FastAPI()
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small and fast
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"  # change if you pulled a different model

# load embedding model once
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# in-memory vector store
vector_store = []  # list of dicts: {'id': int, 'page': int, 'text': str, 'emb': np.array}
emb_matrix = None

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        chunk = text[i : i + max_chars]
        chunks.append(chunk.strip())
        i += max_chars - overlap
    return chunks

def extract_text_from_pdf_bytes(file_bytes: bytes, filename: str) -> List[dict]:
    # returns list of {'page':n, 'text':...}
    results = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text().strip()
        if len(text) < 50:
            # probably scanned image page -> OCR via pdf2image -> pytesseract
            try:
                images = convert_from_path(tmp_path, first_page=page_num+1, last_page=page_num+1, dpi=200)
                if images:
                    text = pytesseract.image_to_string(images[0])
            except Exception as e:
                print("pdf2image error:", e)
        results.append({"page": page_num+1, "text": text})
    try:
        os.unlink(tmp_path)
    except:
        pass
    return results

def add_to_vector_store(chunks_with_meta):
    global vector_store, emb_matrix
    texts = [c["text"] for c in chunks_with_meta]
    if not texts:
        return
    embs = embed_model.encode(texts, convert_to_numpy=True)
    start_id = len(vector_store)
    for i, c in enumerate(chunks_with_meta):
        vector_store.append({
            "id": start_id + i,
            "page": c["page"],
            "text": c["text"],
            "emb": embs[i]
        })
    # rebuild matrix
    emb_matrix = np.vstack([v["emb"] for v in vector_store]) if vector_store else None
    return

def find_similar(query: str, top_k: int = 4):
    if not vector_store:
        return []
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    mat = np.vstack([v["emb"] for v in vector_store])
    # cosine similarity
    dot = mat @ q_emb
    norms = np.linalg.norm(mat, axis=1) * (np.linalg.norm(q_emb) + 1e-12)
    sims = dot / norms
    idx = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        results.append({
            "id": vector_store[i]["id"],
            "page": vector_store[i]["page"],
            "text": vector_store[i]["text"],
            "score": float(sims[i])
        })
    return results

def call_ollama(prompt: str, model: str = OLLAMA_MODEL, max_tokens: int = 300):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        # For /api/chat, the response is in data['message']['content']
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        # Fallback for other possible response structures
        if "response" in data and data["response"]:
            return data["response"]
        if "text" in data:
            return data["text"]
        return json.dumps(data)
    except Exception as e:
        return f"[ERROR calling Ollama]: {e}"

@app.get("/", response_class=HTMLResponse)
def index():
    html = open("static/index.html", "r", encoding="utf-8").read()
    return HTMLResponse(content=html)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    pages = extract_text_from_pdf_bytes(content, file.filename)
    # chunk by page
    chunks = []
    for p in pages:
        for chunk in chunk_text(p["text"], max_chars=900, overlap=200):
            chunks.append({"page": p["page"], "text": chunk})
    add_to_vector_store(chunks)
    return {"status": "ok", "pages": len(pages), "chunks_added": len(chunks)}

@app.post("/query")
async def query(question: str = Form(...), top_k: int = Form(4)):
    sims = find_similar(question, top_k=top_k)
    if not sims:
        return {"answer": "No indexed documents. Upload a PDF first.", "sources": []}
    # build context
    context_texts = []
    for s in sims:
        context_texts.append(f"[page {s['page']}]\n{s['text']}\n")
    context_block = "\n---\n".join(context_texts)
    prompt = (
        "You are an assistant that must answer the user's question using ONLY the information provided in the CONTEXT.\n\n"
        f"CONTEXT:\n{context_block}\n\nQUESTION: {question}\n\n"
        "Produce a concise answer (2-6 sentences). Then list the page numbers you used (e.g., Sources: page 2, page 5). If you can't answer from the context, say you don't know.\n"
    )
    answer = call_ollama(prompt)
    sources = list({s["page"] for s in sims})
    return {"answer": answer, "sources": sources, "matches": sims}
