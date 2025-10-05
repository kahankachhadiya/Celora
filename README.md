# 🧠 Hybrid PDF Processor + RAG Chat UI (Celora)

This repository combines two major components:
1. **`hybrid_converter.py`** — a *Hybrid AI-powered document preprocessor* that converts PDFs into structured, information-rich DOCX files.  
2. **`ui_gradio.py`** — an *interactive Gradio-based UI* that builds a **Retrieval-Augmented Generation (RAG)** chat system to interact with your preprocessed files.

Together, they form a powerful workflow for intelligent document understanding — from raw PDF ingestion to conversational Q&A.

---

## 🚀 Overview

### 🔹 `hybrid_converter.py`

This script performs **hybrid PDF → DOCX conversion** using:
- **Docling AI** → Extracts clean text and tabular data with structure.
- **PyMuPDF (fitz)** → Extracts embedded images from the PDF.
- **PyTorch + CUDA (optional)** → Enables GPU acceleration for faster AI-based text extraction.

The result is a **high-fidelity DOCX file** that preserves text, tables, and images — ideal for downstream NLP or RAG applications.

#### ✅ Key Features
- AI-enhanced extraction of **text**, **tables**, and **images**
- Maintains **1-inch document margins** and layout consistency
- **GPU acceleration** (if CUDA available)
- Auto-cleanup of temporary image files
- Clear console logging for each processing step

---

### 🔹 `ui_gradio.py`

This is a **Gradio-based chat interface** that lets users upload preprocessed files (e.g., DOCX, PDF, TXT, images, audio) and **ask natural language questions** about their content.

It uses the **LangChain** ecosystem to:
- Extract text (from PDFs, DOCX, or OCR on images)
- Transcribe audio (via Whisper)
- Generate embeddings (via HuggingFace)
- Query a **vector database (Chroma)** with a **RetrievalQA** chain
- Optionally perform **vision-based analysis** using an OpenRouter-compatible LLM (e.g., `qwen/qwen-2.5-vl-72b-instruct`)

#### ✅ Key Features
- Multi-modal file ingestion (text, audio, image, PDF)
- RAG-based document querying
- Vision model support for image understanding
- Real-time Gradio UI for interactive chat
- Secure API key handling via environment variable or input box

---

## ⚙️ Workflow

```mermaid
flowchart LR
A[📄 PDF File] --> B[⚙️ hybrid_converter.py\n(AI + PyMuPDF)]
B -->|Outputs| C[📝 DOCX File]
C --> D[🧠 ui_gradio.py\nRAG Chat Interface]
D --> E[💬 User Q&A about content]
