# Local RAG Application (Zero Network Calls)

This project is a fully local, privacy-first Retrieval-Augmented Generation (RAG) application. It leverages **Microsoft Foundry Local** to run language and embedding models entirely on your device without sending any data to the cloud.

## Features
- **Zero Network Calls:** After the initial model download, the application works entirely offline. User data never leaves the device.
- **Cross-Platform:** The Python SDK automatically detects hardware and selects the best execution provider (CPU/GPU/NPU) on Windows, Mac, and Linux.
- **Local Embeddings:** Uses the `qwen3-embedding-0.6b` model to generate document embeddings locally.
- **Local Chat:** Uses the `phi-3.5-mini` model for generating responses based on your context.
- **Serverless Storage:** Utilizes SQLite for storing and retrieving document chunks and vector embeddings seamlessly without extra dependencies.
- **Streamlit Interface:** A clean, minimal web interface for uploading documents and chatting with your data.

## Architecture
- **Runtime:** Microsoft Foundry Local
- **Vector Store:** SQLite (built-in Python `sqlite3` module)
- **Language:** Python
- **UI:** Streamlit

## Setup Instructions

1. **Install Foundry Local CLI:**
   Follow the official Microsoft documentation to install the Foundry Local CLI on your machine.

2. **Download the Models:**
   Open your terminal and run the following commands to cache the models locally:
   ```bash
   foundry model run phi-3.5-mini
   foundry model download qwen3-embedding-0.6b
   ```

3. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   streamlit run main.py
   ```

## Limitations
- Brute-force cosine similarity is used for vector search, which is suitable for smaller-scale document collections.

## License
[MIT](LICENSE)
