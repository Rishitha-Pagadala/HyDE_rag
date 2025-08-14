import streamlit as st
from transformers import pipeline
import numpy as np
import faiss

st.title("HyDE RAG Demo")

# --- Sample FAISS Index ---
embeddings = np.random.rand(5, 384).astype('float32')
index = faiss.IndexFlatL2(384)
index.add(embeddings)

query = st.text_input("Enter your query:")

if query:
    st.write("Searching index...")
    query_vector = np.random.rand(1, 384).astype('float32')
    D, I = index.search(query_vector, k=1)
    st.write(f"Most relevant document ID: {I[0][0]}")

# --- Optional: Use transformers pipeline for text generation ---
generator = pipeline("text-generation", model="distilgpt2")

if st.button("Generate Answer"):
    result = generator(query, max_length=50)
    st.write(result[0]["generated_text"])
