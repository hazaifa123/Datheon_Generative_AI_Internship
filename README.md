# Datheon Generative AI Internship

**Intern:** M. Hazaifa Tahir  
**Repository:** `Datheon_Generative_AI_Internship`

## Completed Tasks

### Task 1: Structured LLM Prompt Pipeline & JSON Extractor
- Extracted structured data from unstructured text using Gemini API and Pydantic validation (`extractor.py`, `output.json`).

### Task 2: Document Q&A Assistant with RAG & Vector Embeddings (`Task_2_RAG_Document_QA/`)
- Built an end-to-end Retrieval-Augmented Generation (RAG) pipeline using **LangChain**, **ChromaDB**, and **Google Gemini**.
- Loads PDF documents, performs recursive text chunking, stores embeddings in ChromaDB, and answers questions with inline source page citations.

### Task 3: Multi-Tool Autonomous AI Agent (`Task_3_Multi_Tool_AI_Agent/`)
- Built an autonomous ReAct agent equipped with **DuckDuckGo Web Search**, a **Safe Math Calculator**, and a **Local File Reader**.
- Streams the step-by-step `Thought -> Action -> Observation -> Final Answer` execution trace (`autonomous_agent.py`, `sample_notes.txt`, `agent_output.txt`).

### Task 4: Interactive Full-Stack AI Chatbot with Streamlit (`Task_4_Streamlit_AI_Chatbot/`)
- Deployed an interactive web-based conversational AI application using **Streamlit**, **LangChain**, and **Google Gemini**.
- Features real-time token streaming (`st.write_stream`), session-based conversation memory (`st.session_state`), and sidebar model controls.
