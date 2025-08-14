import streamlit as st
from utils.document_parser import parse_document
from utils.chunking import chunk_text
from utils.hyde_rag import build_vector_store, hyde_rag_answer
from huggingface_hub import hf_hub_download
import os

if not os.path.exists("./models/tinyllama.gguf"):
    model_path = hf_hub_download(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        local_dir="./models"
    )
st.set_page_config(page_title="HyDE RAG Chatbot", page_icon="🤖", layout="wide")
st.title(" HyDE RAG Chatbot (Upload Document & Ask Questions)")

# Upload document
uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt", "docx"])
if uploaded_file:
    text = parse_document(uploaded_file)
    chunks = chunk_text(text)
    vector_store = build_vector_store(chunks)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    query = st.chat_input("Ask something from your document...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        answer = hyde_rag_answer(query, vector_store, chunks)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
