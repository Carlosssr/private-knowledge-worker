# 🔒 Private Knowledge Worker

A local, private Retrieval-Augmented Generation (RAG) assistant that lets you chat with your own knowledge base using documents and spreadsheets. Built with LangChain, HuggingFace, Chroma DB, and a local LLM (e.g., LLaMA 3 via Ollama) or OpenAI.

---

## 🧠 Features

- 🔐 **Private and Local** – All files stay on your machine.
- 📂 Ingests `.txt`, `.md`, `.csv`, and `.xlsx` from the `/Docs` and `/Spreadsheets` folders.
- 📚 Chunks, embeds, and stores docs in a persistent vector database (Chroma).
- 💬 Uses LangChain’s ConversationalRetrievalChain for context-aware chat.
- 🧠 Memory-enabled chat interface (history preserved during session).
- 🖥️ Built with Gradio for a clean UI experience in your browser.

---

## 🛠️ How to Run

### 1. Clone this repo
```bash
git clone https://github.com/Carlosssr/private-knowledge-worker.git
cd private-knowledge-worker
