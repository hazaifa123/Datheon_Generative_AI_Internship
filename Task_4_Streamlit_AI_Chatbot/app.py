import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Datheon AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Interactive Full-Stack AI Chatbot")
st.caption("Powered by LangChain, Google Gemini & Streamlit | Real-Time Token Streaming")

# ---------------------------------------------------------
# 2. Sidebar Controls & Session State Management
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Enter your Gemini API key or set GOOGLE_API_KEY in .env"
    )

    model_name = st.selectbox(
        "Select Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0
    )

    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1
    )

    system_prompt = st.text_area(
        "System Persona",
        value="You are a helpful, concise, and intelligent AI assistant built with LangChain and Streamlit.",
        height=100
    )

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize conversation history in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------
# 3. Render Existing Chat Message History
# ---------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------
# 4. Handle User Input & Stream AI Response
# ---------------------------------------------------------
if prompt := st.chat_input("Ask me anything..."):
    if not api_key_input:
        st.error("Please enter your Google Gemini API Key in the sidebar to start chatting.")
        st.stop()

    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build LangChain message history for contextual memory
    langchain_messages = [SystemMessage(content=system_prompt)]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))

    # Initialize LangChain Chat Model with streaming enabled
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key_input,
        temperature=temperature,
        streaming=True
    )

    def token_generator():
        for chunk in llm.stream(langchain_messages):
            if chunk.content:
                yield chunk.content

    # Stream assistant response in real time
    with st.chat_message("assistant"):
        try:
            full_response = st.write_stream(token_generator())
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error generating response: {e}")
