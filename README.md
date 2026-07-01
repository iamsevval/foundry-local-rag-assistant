# 🤖 Local Hybrid RAG Assistant (Zero Network Calls)

This project is a fully local, privacy-first Retrieval-Augmented Generation (RAG) application. It leverages **Microsoft Foundry Local SDK** and **sqlite-vec** to run language models and high-performance vector search entirely on your device, ensuring zero data leaves your local environment.

## ✨ Key Features
- **Zero Network Architecture:** After the initial model download, the application works entirely offline. Absolute privacy for your sensitive documents.
- **Advanced Hybrid Search (RRF):** Combines standard semantic Vector Search (`sqlite-vec`) with exact keyword matching via SQLite's Full-Text Search (`FTS5`). Results are intelligently merged using Reciprocal Rank Fusion (RRF).
- **Semantic Chunking:** Documents are split intelligently based on sentence boundaries, rather than blind character counts, ensuring context is never cut in half.
- **Dynamic Persona System:** Instantly change the AI's personality and instructions through the UI sidebar (e.g., "Act as a grumpy software architect").
- **Streaming UI:** Watch the AI type out its answers in real-time (typewriter effect) using a modern **Streamlit** interface.
- **Database & Memory Management:** Includes UI controls to clear the chat history, view loaded documents, and completely wipe the SQLite database with a single click.

## 🏗 Architecture & Stack
- **LLM Runtime:** Microsoft Foundry Local SDK (`phi-3.5-mini`)
- **Embedding Model:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Store:** SQLite with `sqlite-vec` & `FTS5` Virtual Tables
- **Frontend:** Streamlit
- **Language:** Python

## 🚀 Setup Instructions

1. **Install Dependencies:**
   Ensure you are using your virtual environment (`venv`), then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This project relies on `sqlite-vec`, `sentence-transformers`, `streamlit`, and `pypdf`)*

2. **Run the Application:**
   Start the Streamlit application:
   ```bash
   python -m streamlit run app.py
   ```

3. **Using the App:**
   - Upload a PDF or TXT file via the sidebar.
   - Wait for the Hybrid Indexing process to complete.
   - Set a custom Persona if desired.
   - Ask questions and see how the AI retrieves hybrid-scored context!

## 📌 Troubleshooting
- **ModuleNotFoundError:** Ensure you are running the `python -m streamlit` command from *within* your activated `venv`.
- **Foundry SDK Singleton Error:** If you encounter `FoundryLocalException` during Streamlit's Hot Reload, simply refresh the web page (F5). The backend handles this gracefully.

## 📄 License
[MIT](LICENSE)
