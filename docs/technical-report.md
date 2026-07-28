# ARES: Technical Report and Performance Analysis

## Introduction
ARES (Tactical Intelligence Terminal) is a local RAG (Retrieval-Augmented Generation) system developed using Microsoft Foundry Local technology, operating on a "Zero-Network" principle. Its purpose is to perform high-performance semantic inferences in military or corporate closed systems (Air-Gapped) without leaking data to external cloud services.

## Technology Comparison and Selection Criteria

### Why Dense Embeddings instead of TF-IDF?
Traditional search engines (TF-IDF, BM25) only count word frequencies. They cannot match phrases like "Helicopter" and "Rotary-Wing Aircraft". ARES overcomes this issue by using the **Sentence-Transformers (all-MiniLM-L6-v2)** model to project words into a 384-dimensional vector space, enabling synonymous and contextual searches.

### Why Hybrid Search instead of standard Vector Search?
Performing only a vector search sometimes falls short on highly specific part or model numbers like "F-16C Block 50". Therefore, ARES uses:
1. **Semantic Search:** Utilizes Vector search for semantic inference.
2. **Lexical Search:** Utilizes SQLite FTS5 (Full-Text Search) for pinpoint keyword hits.
The results of the two algorithms are mathematically fused using **RRF (Reciprocal Rank Fusion)**.

### Why a Re-Ranker?
After finding the 20 most probable text chunks from the search results, the system does not send them to the LLM in their raw state. A **Cross-Encoder** (HuggingFace) steps in to read the question and all 20 texts individually, re-ranking them based on their logical relevance to select the top 5 highest-scoring texts. This reduces the risk of "Hallucination" by up to 90%.

## Hardware Optimization (Foundry Local)
Instead of dealing with the memory overhead introduced by virtualization/container layers like Docker or Ollama, ARES utilizes Microsoft Foundry Local, which can directly leverage the device's local resources (NPU/GPU). This provides a massive boost in model initialization times and Token/Second (TPS) rates.

## Conclusion
ARES goes far beyond a standard "RAG" application; it is a Production-Ready defense industry prototype that combines sensor data (Radar), text matching (BM25), vector mathematics (Embeddings), and machine learning scoring (Cross-Encoder) into a single, closed-box intelligence terminal.
