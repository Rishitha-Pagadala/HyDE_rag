import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Load models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = Llama(model_path="./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", n_ctx=2048)

def build_vector_store(chunks):
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def hyde_rag_answer(query, vector_store, chunks, top_k=3):
    # 1. Generate hypothetical answer
    prompt = f"Answer this question based only on the uploaded document:\n\n{query}\n\nAnswer:"
    hypo_answer = llm(prompt, max_tokens=200)["choices"][0]["text"].strip()

    # 2. Embed hypothetical answer
    query_emb = embed_model.encode([hypo_answer])

    # 3. Retrieve top chunks
    D, I = vector_store.search(np.array(query_emb).astype('float32'), k=top_k)
    retrieved = [chunks[i] for i in I[0]]

    # 4. Combine retrieved chunks
    return "\n\n".join(retrieved)
