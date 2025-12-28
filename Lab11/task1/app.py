import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss #pip install faiss-cpu
from PyPDF2 import PdfReader
import numpy as np
from gpt4all import GPT4All

# =========================
# APP TITLE
# =========================
st.title("Offline Free RAG PDF Reader 📄")
st.write("""
Upload a PDF and ask questions.
This app uses **local embeddings + FAISS retrieval + local GPT4All LLM**.
""")

# =========================
# PDF UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # =========================
    # READ PDF
    # =========================
    pdf = PdfReader(uploaded_file)
    full_text = ""

    for page in pdf.pages:
        text = page.extract_text()
        if text:
            full_text += text + " "

    # =========================
    # SIMPLE CHUNKING
    # =========================
    chunks = [
        chunk.strip()
        for chunk in full_text.split(".")
        if len(chunk.strip()) > 30
    ]

    st.success(f"PDF loaded successfully! {len(chunks)} text chunks created.")

    # =========================
    # LOAD EMBEDDING MODEL
    # =========================
    @st.cache_resource
    def load_embedding_model():
        return SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    embed_model = load_embedding_model()

    # =========================
    # CREATE EMBEDDINGS
    # =========================
    embeddings = embed_model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")

    # =========================
    # FAISS INDEX
    # =========================
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # =========================
    # LOAD LOCAL GPT4ALL MODEL
    # =========================
    @st.cache_resource
    def load_llm():
        model_dir = r"E:\University\7th Semester\ML\Lab\Lab11\task1"
        model_name = "q4_0-orca-mini-3b.gguf"

        return GPT4All(
            model_name=model_name,
            model_path=model_dir,
            allow_download=False
        )

    llm = load_llm()

    # =========================
    # USER QUESTION
    # =========================
    query = st.text_input("Ask a question from the PDF:")

    if query:
        # =========================
        # EMBED QUERY
        # =========================
        query_embedding = embed_model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        # =========================
        # RETRIEVE TOP-K CHUNKS
        # =========================
        _, indices = index.search(query_embedding, k=3)
        retrieved_chunks = [chunks[i] for i in indices[0]]

        st.subheader("🔍 Retrieved Context")
        for i, chunk in enumerate(retrieved_chunks, start=1):
            st.write(f"{i}. {chunk}")

        # =========================
        # RAG PROMPT
        # =========================
        prompt = f"""
Answer the question using ONLY the context below.

Context:
{chr(10).join(retrieved_chunks)}

Question:
{query}

Answer:
"""

        # =========================
        # GENERATE ANSWER
        # =========================
        with st.spinner("Generating answer..."):
            answer = llm.generate(prompt, max_tokens=300)

        st.subheader("🧠 Generated Answer")
        st.write(answer)