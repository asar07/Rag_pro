import streamlit as st
import os
import tempfile
import requests
import faiss
import numpy as np
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from sentence_transformers import SentenceTransformer
from bytez import Bytez

st.set_page_config(page_title="DocChat Pro", page_icon="📄", layout="centered")

# ---------------- SETTINGS ----------------

MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku-20240307",
    "meta-llama/Llama-3.1-8B-Instruct",
]

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------- API KEY ----------------

def get_api_key():
    try:
        return st.secrets["BYTEZ_API_KEY"]
    except:
        return None


# ---------------- TEXT EXTRACTORS ----------------

def extract_pdf(file_bytes):

    pages = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    reader = PdfReader(path)

    for i, page in enumerate(reader.pages):

        text = page.extract_text() or ""

        if text.strip():

            pages.append({
                "page": i + 1,
                "text": text
            })

    return pages


def extract_docx(file_bytes):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    doc = Document(path)

    text = "\n".join(p.text for p in doc.paragraphs)

    return [{"page": 1, "text": text}]


def extract_pptx(file_bytes):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    prs = Presentation(path)

    slides = []

    for i, slide in enumerate(prs.slides):

        text = []

        for shape in slide.shapes:

            if hasattr(shape, "text"):
                text.append(shape.text)

        slides.append({
            "page": i + 1,
            "text": "\n".join(text)
        })

    return slides


def extract_txt(file_bytes):

    text = file_bytes.decode()

    return [{"page": 1, "text": text}]


def extract_url(url):

    html = requests.get(url).text

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text()

    return [{"page": 1, "text": text}]


# ---------------- CHUNKING ----------------

def chunk_text(pages, size=400, overlap=80):

    chunks = []

    for p in pages:

        words = p["text"].split()

        start = 0

        while start < len(words):

            end = min(start + size, len(words))

            chunks.append({
                "text": " ".join(words[start:end]),
                "page": p["page"]
            })

            start += size - overlap

    return chunks


# ---------------- EMBEDDINGS ----------------

@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)


def embed_texts(texts):

    model = load_embed_model()

    return model.encode(texts)


# ---------------- FAISS INDEX ----------------

def build_faiss_index(chunks):

    texts = [c["text"] for c in chunks]

    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings).astype("float32"))

    return index, embeddings


# ---------------- RETRIEVAL ----------------

def retrieve(query, chunks, index, k=4):

    model = load_embed_model()

    qvec = model.encode([query]).astype("float32")

    D, I = index.search(qvec, k)

    results = []

    for i in I[0]:
        results.append(chunks[i])

    return results


# ---------------- MODEL ANSWER ----------------

def ask(question, context_chunks, api_key, model_id):

    context = "\n\n".join(
        f"(Page {c['page']}) {c['text']}" for c in context_chunks
    )

    system_prompt = f"""
You are a document QA assistant.

Rules:
- Answer ONLY using the provided document context
- If information is missing say "The document does not contain that information"
- Cite page numbers like (Page 2)
- Be concise

Document context:
{context}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    try:

        sdk = Bytez(api_key)

        model = sdk.model(model_id)

        result = model.run(messages)

        if result.error:
            return result.error

        return str(result.output)

    except Exception as e:
        return str(e)


# ---------------- SESSION ----------------

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------- UI ----------------

st.title("📄 DocChat Pro")

api_key = get_api_key()

if not api_key:
    api_key = st.text_input("Bytez API Key", type="password")

uploaded = st.file_uploader(
    "Upload document",
    type=["pdf", "docx", "pptx", "txt"]
)

url_input = st.text_input("Or enter URL")


with st.expander("Settings"):

    model_id = st.selectbox("Model", MODELS)

    chunk_size = st.slider("Chunk Size", 200, 800, 400)

    overlap = st.slider("Overlap", 0, 150, 80)

    top_k = st.slider("Top Results", 2, 8, 4)


# ---------------- PROCESS DOCUMENT ----------------

if uploaded:

    with st.spinner("Processing document..."):

        file_bytes = uploaded.read()

        if uploaded.name.endswith(".pdf"):
            pages = extract_pdf(file_bytes)

        elif uploaded.name.endswith(".docx"):
            pages = extract_docx(file_bytes)

        elif uploaded.name.endswith(".pptx"):
            pages = extract_pptx(file_bytes)

        else:
            pages = extract_txt(file_bytes)

        chunks = chunk_text(pages, chunk_size, overlap)

        index, embeddings = build_faiss_index(chunks)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.history = []

        st.success("Document indexed")


if url_input:

    with st.spinner("Fetching URL..."):

        pages = extract_url(url_input)

        chunks = chunk_text(pages)

        index, embeddings = build_faiss_index(chunks)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.history = []


# ---------------- CHAT ----------------

if st.session_state.chunks:

    for msg in st.session_state.history:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about your document")

    if prompt:

        st.session_state.history.append({
            "role": "user",
            "content": prompt
        })

        with st.spinner("Thinking..."):

            sources = retrieve(
                prompt,
                st.session_state.chunks,
                st.session_state.index,
                top_k
            )

            answer = ask(
                prompt,
                sources,
                api_key,
                model_id
            )

            pages = sorted(set(c["page"] for c in sources))

            citation = f"\n\nSources: {', '.join('Page '+str(p) for p in pages)}"

            final = answer + citation

        st.session_state.history.append({
            "role": "assistant",
            "content": final
        })

        st.rerun()
