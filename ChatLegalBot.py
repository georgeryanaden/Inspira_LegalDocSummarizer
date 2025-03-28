import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import torch
import tempfile
import os
from typing import List

# Load models
st.session_state.setdefault("embedder", SentenceTransformer("all-MiniLM-L6-v2"))
st.session_state.setdefault("qa_model_name", "google/flan-t5-base")
st.session_state.setdefault("tokenizer", AutoTokenizer.from_pretrained(st.session_state.qa_model_name))
st.session_state.setdefault("model", AutoModelForSeq2SeqLM.from_pretrained(st.session_state.qa_model_name))
st.session_state.setdefault("summarizer", pipeline("text2text-generation", model=st.session_state.model, tokenizer=st.session_state.tokenizer))

# Helper functions
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_chunks(chunks: List[str]):
    return st.session_state.embedder.encode(chunks)

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(query, chunks, index, chunk_embeddings, top_k=3):
    query_embedding = st.session_state.embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def answer_question(query, context_chunks):
    context = "\n\n".join(context_chunks)
    max_input_tokens = 512
    tokenizer = st.session_state.tokenizer
    tokens = tokenizer.encode(context + query, truncation=True, max_length=max_input_tokens)
    input_str = tokenizer.decode(tokens, skip_special_tokens=True)

    prompt = f"Answer the question using only the context below:\n\n{input_str}\n\nQuestion: {query}"
    result = st.session_state.summarizer(prompt, max_length=256, do_sample=False)[0]

    # Use fallback key if needed
    return result.get('summary_text') or result.get('generated_text') or "⚠️ Could not generate a response."

# Streamlit UI
st.set_page_config(page_title="Chat with a Contract")
st.title("🤖 Chat with a Contract")
st.markdown("Upload a legal PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload a contract PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting and chunking text..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        chunk_embeddings = embed_chunks(chunks)
        index = create_faiss_index(chunk_embeddings)

    st.success("Contract processed! You can now ask questions.")

    user_question = st.text_input("Ask a question about the contract:")
    if user_question:
        with st.spinner("Thinking..."):
            relevant_chunks = retrieve_relevant_chunks(user_question, chunks, index, chunk_embeddings)
            answer = answer_question(user_question, relevant_chunks)
        st.markdown("### 📌 Answer")
        st.write(answer)

        with st.expander("🔍 Context used"):
            for i, chunk in enumerate(relevant_chunks):
                st.markdown(f"**Chunk {i+1}:**")
                st.code(chunk)
else:
    st.info("Please upload a contract to get started.")