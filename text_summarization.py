import os
# Set environment variables before any imports
os.environ["STREAMLIT_DISABLE_WATCHDOG"] = "true"
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

import uuid
import tempfile
import nltk
import time
import streamlit as st
import requests
import chromadb
from chromadb.utils import embedding_functions
from nltk.tokenize import sent_tokenize
from langchain_community.document_loaders import PyMuPDFLoader

# ===== Download NLTK Dependencies =====
nltk.download('punkt', quiet=True)

# ===== Configuration =====
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_SUMMARY_MODEL = "tinyllama:latest"
OLLAMA_API_URL = "http://localhost:11434"
OLLAMA_EMBED_API_URL = f"{OLLAMA_API_URL}/api/embeddings"
OLLAMA_GENERATE_API_URL = f"{OLLAMA_API_URL}/api/generate"
CHUNK_SIZE = 500  # Optimized for low latency
TOP_K = 4  # Optimized for fast retrieval

# ===== Cache Dependencies =====
@st.cache_resource(show_spinner=False)
def load_dependencies():
    ollama_embedding = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_EMBED_API_URL,
        model_name=OLLAMA_EMBED_MODEL
    )
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="pdf_chunks",
        embedding_function=ollama_embedding
    )
    return collection

collection = load_dependencies()

# ===== Chunk PDF =====
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

# ===== Store in ChromaDB with Batch Processing =====
def store_chunks(collection, chunks, pdf_hash):
    existing_ids = collection.get()['ids']
    if existing_ids:
        collection.delete(ids=existing_ids)
    
    # Batch processing to reduce API calls
    batch_size = 50  # Adjust based on system capacity
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            documents=[chunk["content"] for chunk in batch],
            ids=[f"{pdf_hash}_{uuid.uuid4()}" for _ in batch],
            metadatas=[chunk["metadata"] for chunk in batch]
        )

# ===== Retrieve Relevant Chunks =====
def retrieve_context(collection, query):
    results = collection.query(query_texts=[query], n_results=TOP_K)
    documents = results.get('documents', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]
    distances = results.get('distances', [[]])[0]

    if not documents:
        return [], [], []

    scored = list(zip(documents, distances, metadatas))
    scored.sort(key=lambda x: x[1])
    top_chunks = [doc for doc, _, _ in scored[:TOP_K]]
    top_distances = [distance for _, distance, _ in scored[:TOP_K]]
    top_meta = [meta for _, _, meta in scored[:TOP_K]]

    return top_chunks, top_distances, top_meta

# ===== Summarization Logic with TinyLLaMA =====
def summarize_text(text):
    prompt = f"Generate a concise summary (20-50 words) of the following text. Focus on the main ideas, omit minor details, and avoid repetition. Ensure clarity and relevance:\n\n{text}"
    payload = {
        "model": OLLAMA_SUMMARY_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 50,
            "temperature": 0.7,
        }
    }
    try:
        response = requests.post(OLLAMA_GENERATE_API_URL, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "Summary not available.").strip()
    except requests.RequestException as e:
        return f"Error summarizing text: {str(e)}"

# ===== Streamlit UI =====
if __name__ == "__main__":
    st.set_page_config(page_title="PDF Chatbot with Summarization", page_icon="📄")
    st.title("📄 Precision PDF Q&A with Summarization")

    # Initialize session state
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
        st.session_state.pdf_hash = None
        st.session_state.chunk_count = 0

    pdf_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

    if pdf_file:
        # Generate a unique hash for the PDF based on its content
        pdf_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, pdf_file.getvalue().hex()))
        
        # Process PDF only if it hasn't been processed or is different
        if not st.session_state.pdf_processed or st.session_state.pdf_hash != pdf_hash:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                tmp.flush()  # Ensure file is written
                pdf_path = tmp.name

            try:
                with st.spinner("🔍 Processing and embedding document..."):
                    chunks = load_pdf_chunks(pdf_path)
                    store_chunks(collection, chunks, pdf_hash)
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_hash = pdf_hash
                    st.session_state.chunk_count = len(chunks)

                st.success(f"✅ Document indexed with {st.session_state.chunk_count} chunks.")
            finally:
                os.unlink(pdf_path)  # Clean up temporary file

        else:
            st.success(f"✅ Document already indexed with {st.session_state.chunk_count} chunks.")

        question = st.text_input("🔎 Enter your question about the PDF:")

        if st.button("💬 Generate Answer") and question.strip():
            start_time = time.time()

            with st.spinner("📅 Retrieving relevant context..."):
                top_chunks, distances, metas = retrieve_context(collection, question)

            latency = time.time() - start_time
            st.markdown(f"⏱ *Latency:* {latency:.2f} seconds")

            if not top_chunks:
                st.warning("⚠ No relevant information found. Try rephrasing the question.")
            else:
                top_chunk = top_chunks[0]
                top_distance = distances[0]
                top_meta = metas[0]

                st.markdown("### ✅ Answer extracted from the chunks")
                st.markdown(f"*Page:* {top_meta['page']}  |  *Distance:* {top_distance:.4f}")
                st.code(top_chunk, language="markdown")

                st.markdown("### 📝 Text summarization for that topic")
                chunk_summary = summarize_text(top_chunk)
                st.success(chunk_summary)

                if len(top_chunks) > 1:
                    with st.expander("📌 Other Top Supporting Chunks"):
                        for i, (chunk, distance, meta) in enumerate(zip(top_chunks[1:], distances[1:], metas[1:])):
                            st.markdown(f"*Chunk {i+2}* | Page: {meta['page']} | Distance: {distance:.4f}")
                            st.code(chunk, language="markdown")
    else:
        st.info("📄 Please upload a PDF to begin.")
