# LLM-PDF-QNA
PDF Question Answering App

This project is a **PDF Q&A application** built with **FastAPI** that allows users to upload PDF documents and ask natural language questions about their content.  

It combines:
- **OCR** for scanned PDFs (`pytesseract`, `pdf2image`, `Pillow`)
- **NLP embeddings** with `sentence-transformers`
- **API-first architecture** powered by **FastAPI**
- **Vector-based semantic search** to find the most relevant answers

---

## 🚀 Features
- Upload PDF files (text-based or scanned)
- Extracts and processes text automatically
- Handles scanned PDFs using OCR (`pytesseract`)
- Embeds text with `sentence-transformers` for semantic similarity
- Query the document and get precise answers
- FastAPI backend with auto-generated docs (`/docs`)

---

## 🛠️ Tech Stack
- **Backend Framework:** [FastAPI](https://fastapi.tiangolo.com/)  
- **Language:** Python 3.x  
- **NLP & Embeddings:** `sentence-transformers`  
- **OCR & PDF Processing:** `pytesseract`, `pdf2image`, `Pillow`  
- **API Serving:** `uvicorn`  
- **Data Handling:** `numpy`, `requests`, `python-multipart`

---

## 📂 Project Structure
