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

1. **Run `hybrid_converter.py`** to preprocess and convert PDFs into high-quality `.docx` files.  
2. **Upload** the resulting DOCX (or other supported files) into the **Gradio interface**.  
3. **Chat** with the content using natural language questions.

---

## 🧩 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hybrid-pdf-rag.git
cd hybrid-pdf-rag

# Install dependencies
pip install -r requirements.txt
```

> 🧠 Make sure you have [CUDA](https://developer.nvidia.com/cuda-downloads) installed for GPU acceleration (optional but recommended).

---

## 🔑 Environment Variables

You can configure API and model settings using environment variables:

```bash
export OPENAI_API_KEY="sk-or-xxxxxxxxxxxx"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
export PERSIST_DIR="./db"
export VISION_MODEL="qwen/qwen-2.5-vl-72b-instruct"
```

---

## 🧠 Usage

### 1️⃣ Run Hybrid Preprocessor

```bash
python hybrid_converter.py
```

- Converts `sample.pdf` → `sample_hybrid.docx`
- Extracts images, tables, and formatted text

---

### 2️⃣ Launch Chat UI

```bash
python ui_gradio.py
```

- Opens a Gradio web app at `http://localhost:7860`
- Upload the DOCX (or other supported files)
- Enter your OpenRouter API key
- Ask questions like:

```text
"Summarize this document."
"List all tables mentioned."
"What does the chart on page 3 represent?"
```

---

## 🧱 Project Structure

```
📦 hybrid-pdf-rag/
│
├── hybrid_converter.py    # Hybrid PDF → DOCX converter
├── ui_gradio.py           # Gradio-based RAG chat interface
├── requirements.txt       # Dependencies
└── README.md              # You are here
```

---

## 💡 Example Use Cases

- Research papers → Extract structured DOCX → Query specific sections  
- Financial reports → Extract tables & figures → Ask performance questions  
- Scanned PDFs → OCR & summarize via RAG pipeline  

---

## 🧰 Requirements

- **Python** ≥ 3.9  
- **Libraries**:
  - `torch`, `fitz` (PyMuPDF), `docx`, `docling`
  - `langchain`, `gradio`, `pytesseract`, `whisper`
  - `sentence-transformers`, `chromadb`

---

## 🧾 License

**MIT License © 2025 Celora**

---

> 💬 *“AI-enhanced document understanding — from PDF to conversation.”*
