# ARES Tactical Intelligence Terminal - System Architecture

This document outlines the technical infrastructure and data flow architecture of the ARES system. The system design is built upon operating in **"Zero-Network" (Air-Gapped)** environments ensuring maximum data security, while fusing real-time telemetry data from external sensors with local artificial intelligence (Sensor Fusion).

## 1. Core Components

The system architecture consists of 3 main layers:

### A. Presentation Layer
- **Technology:** Streamlit & HTML/CSS Injection
- **Role:** Serves as the "Tactical HUD (Heads-Up Display)" interface where the user interacts with the system.
- **File:** `app.py`

### B. Microservice & Integration Layer (API)
- **Technology:** FastAPI, Uvicorn, Python Requests
- **Role:** 
  1. Provides a Headless REST API (`api.py`) allowing the system to communicate with other software (e.g., UAV flight computers).
  2. Fetches real-time radar and flight telemetry from external sources (OpenSky ADS-B Network) via `src/radar_sensor.py`.

### C. AI Engine & Storage Layer
- **Technology:** Microsoft Foundry Local, Sentence-Transformers (all-MiniLM-L6-v2, Cross-Encoder), SQLite (FTS5 + sqlite-vec)
- **Role:** Handles the semantic chunking of documents (`src/document_processor.py`), saving to the vector database (`src/vector_db.py`), and answer generation via the local LLM (Phi-3.5) (`src/rag_core.py`).

## 2. Sensor Fusion Data Flow

The radar integration, which is the most innovative aspect of the system, works as follows:
1. When the user activates the radar from the interface, the `src/radar_sensor.py` module connects to the OpenSky network.
2. Altitude (m) and Velocity (m/s) data are fetched and updated continuously.
3. When the user asks a question, `src/rag_core.py` dynamically injects this real-time sensor data into the LLM's (Phi-3.5) "System Prompt" context.
4. The LLM fuses the military documents (Flight Manuals) fetched from the database with the real-time flight telemetry to generate immediate tactical decisions for the headquarters.

## 3. Hybrid Search and Graph-RAG

The system goes beyond simple vector similarity:
- **Lexical Search (BM25):** Traditional keyword matching (SQLite FTS5).
- **Semantic Search:** Semantic proximity of sentences.
- **RRF (Reciprocal Rank Fusion):** Mathematical combination of both search results.
- **Cross-Encoder Re-Ranking:** Re-scoring the logical relevance of the retrieved documents to the actual question, heavily reducing hallucinations.

Furthermore, every uploaded document undergoes entity-relationship analysis using NLP techniques and is transformed into a 3D **Knowledge Graph** via PyVis.

## 4. Security and Hardware
Thanks to the Microsoft Foundry Local infrastructure, no requests are sent to cloud services or remote servers (OpenAI, Azure, etc.) for the Phi-3.5 model. All model weights are kept on the local disk and executed directly on the device's NPU/GPU hardware (On-Device). This architectural design is 100% compliant with military "Air-Gapped" systems.
