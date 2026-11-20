# Task 4: Interactive Full-Stack AI Chatbot with Streamlit

**Intern:** M. Hazaifa Tahir  
**Program:** Datheon Generative AI Internship  

## Overview
A responsive, full-stack conversational AI web application built using **Streamlit**, **LangChain**, and **Google Gemini**. It supports real-time token streaming, persistent multi-turn conversation memory via `st.session_state`, customizable model parameters, and one-click chat reset.

## Key Features
- **Real-Time Token Streaming:** Uses `llm.stream()` paired with `st.write_stream()` for smooth token-by-token responses.
- **Conversation Memory:** Maintains full multi-turn chat history using `st.session_state` and LangChain message objects (`SystemMessage`, `HumanMessage`, `AIMessage`).
- **Interactive UI Controls:** Sidebar configuration for API key input, model selection (`gemini-2.5-flash` / `gemini-2.5-pro`), temperature slider, custom system prompt, and clearing chat history.

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
