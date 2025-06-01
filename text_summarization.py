import streamlit as st
import requests
import uuid
import tempfile
import json
import nltk
import time
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyMuPDFLoader
import chromadb
from chromadb.utils import embedding_functions
from nltk.tokenize import sent_tokenize

# Download nltk tokenizer models
nltk.download('punkt')

# ===== Configuration =====
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_API_URL = "http://localhost:11434/api/embeddings"

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 800
TOP_K = 6
RERANK_TOP_K = 3
SCORE_THRESHOLD = 0.3

# ===== Cache Dependencies =====
@st.cache_resource(show_spinner=False)
def load_dependencies():
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    ollama_embedding = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_EMBED_API_URL,
        model_name=OLLAMA_EMBED_MODEL
    )
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="pdf_chunks",
        embedding_function=ollama_embedding
    )
    return cross_encoder, collection

cross_encoder, collection = load_dependencies()

# ===== PDF Chunk Loader =====
def load_pdf_chunks(path):
    loader = PyMuPDFLoader(path)
    docs = loader.load()
    chunks = []

    for i, doc in enumerate(docs):
        sentences = sent_tokenize(doc.page_content)
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) <= CHUNK_SIZE:
                current_chunk += " " + sent
            else:
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": {"page": i + 1}
                })
                current_chunk = sent
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": {"page": i + 1}
            })
    return chunks

# ===== Store chunks in ChromaDB =====
def store_chunks(chunks):
    existing_ids = collection.get()['ids']
    if existing_ids:
        collection.delete(ids=existing_ids)
    for chunk in chunks:
        collection.add(
            documents=[chunk["content"]],
            ids=[str(uuid.uuid4())],
            metadatas=[chunk["metadata"]]
        )

# ===== Retrieve and rerank relevant chunks =====
def retrieve_context(query):
    results = collection.query(query_texts=[query], n_results=TOP_K)
    documents = results.get('documents', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]
    ids = results.get('ids', [[]])[0]

    pairs = [[query, doc] for doc in documents]
    scores = cross_encoder.predict(pairs)

    scored = list(zip(documents, scores, metadatas, ids))
    scored.sort(key=lambda x: x[1], reverse=True)
    filtered = [(doc, score, meta) for doc, score, meta, _ in scored if score >= SCORE_THRESHOLD]

    top_chunks = [doc for doc, _, _ in filtered[:RERANK_TOP_K]]
    top_scores = [score for _, score, _ in filtered[:RERANK_TOP_K]]
    top_meta = [meta for _, _, meta in filtered[:RERANK_TOP_K]]

    return top_chunks, top_scores, top_meta

# ===== Streamlit UI =====
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Precision PDF Q&A (Extractive, Top Chunk Only)")

pdf_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name

    with st.spinner("🔍 Processing and embedding document..."):
        chunks = load_pdf_chunks(pdf_path)
        store_chunks(chunks)

    st.success(f"✅ Document indexed with {len(chunks)} chunks.")

    question = st.text_input("🔎 Enter your question about the PDF:")

    if st.button("💬 Generate Answer") and question.strip():
        start_time = time.time()

        with st.spinner("📥 Retrieving and reranking context..."):
            top_chunks, scores, metas = retrieve_context(question)

        latency = time.time() - start_time

        st.markdown(f"⏱️ **Latency:** {latency:.2f} seconds")

        if not top_chunks:
            st.warning("⚠️ No relevant information found. Try rephrasing the question.")
        else:
            top_chunk = top_chunks[0]
            top_score = scores[0]
            top_meta = metas[0]

            st.markdown("### ✅ Answer (Extracted from Top Chunk)")
            st.markdown(f"**Page:** {top_meta['page']}  |  **Score:** {top_score:.4f}")
            st.code(top_chunk, language="markdown")

            if len(top_chunks) > 1:
                with st.expander("📌 Other Top Supporting Chunks"):
                    for i, (chunk, score, meta) in enumerate(zip(top_chunks[1:], scores[1:], metas[1:])):
                        st.markdown(f"**Chunk {i+2}** | Page: `{meta['page']}` | Score: `{score:.4f}`")
                        st.code(chunk, language="markdown")
else:
    st.info("📄 Please upload a PDF to begin.")
