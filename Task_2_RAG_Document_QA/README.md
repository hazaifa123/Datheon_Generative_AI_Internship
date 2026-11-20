# Task 2: Document Q&A Assistant with RAG & Vector Embeddings

**Intern:** M. Hazaifa Tahir  
**Program:** Datheon Generative AI Internship  

## Overview
A Retrieval-Augmented Generation (RAG) pipeline built using **LangChain**, **ChromaDB**, and **Google Gemini** that answers user queries strictly from custom PDF documents along with accurate source page citations.

## Architecture & Workflow
1. **Document Ingestion (`PyPDFLoader`):** Loads multi-page PDF documents while preserving page-level metadata.
2. **Recursive Chunking (`RecursiveCharacterTextSplitter`):** Splits extracted text into overlapping chunks (`chunk_size=1000`, `chunk_overlap=200`) to retain semantic context across boundaries.
3. **Vector Embeddings & Storage (`ChromaDB`):** Converts text chunks into dense vector embeddings using `models/text-embedding-004` and persists them in a local Chroma vector store.
4. **Grounded Retrieval & Generation:** Retrieves the top-4 most relevant chunks via similarity search and enforces strict grounding via prompt instructions so answers include `(Page X)` citations without hallucination.

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
