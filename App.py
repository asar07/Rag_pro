import streamlit as st
import os
import tempfile
import math
import requests
import json

from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from bs4 import BeautifulSoup

st.set_page_config(page_title="DocChat", page_icon="📄", layout="centered")


# ---------------- API KEY ----------------

def get_api_key():
    try:
        return st.secrets["BYTEZ_API_KEY"]
    except Exception:
        return None


# ---------------- MODELS ----------------

MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-haiku-20240307",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


# ---------------- CLEAN MODEL OUTPUT ----------------

def clean_output(raw):
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "content" in parsed:
                return parsed["content"]
        except Exception:
            pass
        if "'content':" in raw:
            try:
                return raw.split("'content':")[1].split("'}")[0].strip(" '")
            except Exception:
                pass
        return raw

    if isinstance(raw, dict):
        if "content" in raw:
            return raw["content"]
        if "generated_text" in raw:
            return raw["generated_text"]

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                if "message" in item:
                    return item["message"]["content"]
                if "generated_text" in item:
                    return item["generated_text"]

    return str(raw)


# ---------------- PDF ----------------

def extract_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    pages = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
        return pages
    finally:
        os.unlink(path)


# ---------------- DOCX ----------------

def extract_docx(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        doc = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"page": 1, "text": text}]
    finally:
        os.unlink(path)


# ---------------- PPTX ----------------

def extract_pptx(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    slides = []
    try:
        prs = Presentation(path)
        for i, slide in enumerate(prs.slides):
            text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
            content = "\n".join(text)
            if content.strip():
                slides.append({"page": i + 1, "text": content})
        return slides
    finally:
        os.unlink(path)


# ---------------- TXT ----------------

def extract_txt(file_bytes):
    text = file_bytes.decode("utf-8", errors="ignore")
    return [{"page": 1, "text": text}]


# ---------------- URL ----------------

def extract_url(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.extract()
        text = soup.get_text(separator=" ")
        return [{"page": 1, "text": text}]
    except Exception:
        return [{"page": 1, "text": ""}]


# ---------------- CHUNKING ----------------

def chunk_pages(pages, size=400, overlap=60):
    chunks = []
    for p in pages:
        words = p["text"].split()
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            chunks.append({
                "text": " ".join(words[start:end]),
                "page": p["page"],
                "id": len(chunks)
            })
            if end == len(words):
                break
            start += size - overlap
    return chunks


# ---------------- VECTOR INDEX ----------------

def build_index(chunks):
    vocab = {}
    for c in chunks:
        for w in c["text"].lower().split():
            if w not in vocab:
                vocab[w] = len(vocab)

    N = len(chunks)
    df = [0] * len(vocab)
    for c in chunks:
        for w in set(c["text"].lower().split()):
            if w in vocab:
                df[vocab[w]] += 1

    idf = [math.log((N + 1) / (d + 1)) + 1 for d in df]

    def vec(text):
        words = text.lower().split()
        tf = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        n = len(words) or 1
        v = [0.0] * len(vocab)
        for w, cnt in tf.items():
            if w in vocab:
                v[vocab[w]] = (cnt / n) * idf[vocab[w]]
        return v

    vecs = [vec(c["text"]) for c in chunks]
    return vocab, vecs, vec


# ---------------- COSINE ----------------

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-9)


# ---------------- RETRIEVE ----------------

def retrieve(query, chunks, vecs, vec_fn, k=4):
    qv = vec_fn(query)
    scored = sorted(
        enumerate(vecs),
        key=lambda x: cosine(qv, x[1]),
        reverse=True
    )
    return [chunks[i] for i, _ in scored[:k]]


# ---------------- MODEL CALL (Direct REST API - no SDK needed) ----------------

def ask(question, context_chunks, history, api_key, model_id):
    context = "\n\n".join(
        f"(Page {c['page']}) {c['text']}"
        for c in context_chunks
    )

    system_prompt = f"""You are a document analysis assistant.

Rules:
1. Answer ONLY using the provided context.
2. If the answer is missing say: "The document does not contain that information."
3. Cite page numbers when possible.
4. Be concise.
5. Use bullet points for lists.

DOCUMENT CONTEXT:
{context}
"""

    messages = []
    for m in history[-6:]:
        messages.append(m)
    messages.append({"role": "user", "content": question})

    try:
        response = requests.post(
            "https://api.bytez.com/models/v2/chat",
            headers={
                "Authorization": f"Key {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "messages": [{"role": "system", "content": system_prompt}] + messages,
                "max_tokens": 1024,
            },
            timeout=60,
        )

        if response.status_code != 200:
            return f"API error {response.status_code}: {response.text}"

        data = response.json()

        # Handle standard OpenAI-compatible response shape
        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        # Fallback to clean_output for other shapes
        return clean_output(data)

    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"API error: {str(e)}"


# ---------------- SESSION STATE ----------------

for key, default in [
    ("chunks", []),
    ("vecs", []),
    ("vec_fn", None),
    ("history", []),
    ("doc_loaded", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------- UI ----------------

st.title("📄 Multi-Document Chat")

secret_key = get_api_key()
if secret_key:
    api_key = secret_key
    st.success("✅ API key loaded from secrets", icon="🔑")
else:
    api_key = st.text_input("Bytez API Key", type="password")

url = st.text_input("Load from URL")
uploaded = st.file_uploader(
    "Upload Document",
    type=["pdf", "docx", "pptx", "txt"]
)

with st.expander("⚙️ Settings"):
    model_id = st.selectbox("Model", MODELS)
    chunk_size = st.slider("Chunk Size", 200, 800, 400)
    overlap = st.slider("Overlap", 0, 150, 60)
    top_k = st.slider("Top Chunks", 2, 8, 4)


# ---------------- LOAD DOCUMENT ----------------

pages = None

if uploaded:
    file_bytes = uploaded.read()
    name = uploaded.name.lower()
    with st.spinner("Extracting document..."):
        if name.endswith(".pdf"):
            pages = extract_pdf(file_bytes)
        elif name.endswith(".docx"):
            pages = extract_docx(file_bytes)
        elif name.endswith(".pptx"):
            pages = extract_pptx(file_bytes)
        elif name.endswith(".txt"):
            pages = extract_txt(file_bytes)

elif url:
    with st.spinner("Fetching URL..."):
        pages = extract_url(url)

if pages:
    chunks = chunk_pages(pages, chunk_size, overlap)
    if chunks:
        vocab, vecs, vec_fn = build_index(chunks)
        st.session_state.chunks = chunks
        st.session_state.vecs = vecs
        st.session_state.vec_fn = vec_fn
        st.session_state.history = []  # Reset chat on new doc
        st.success(f"✅ Loaded {len(pages)} page(s) → {len(chunks)} chunks")
    else:
        st.warning("No text could be extracted from the document.")


# ---------------- CHAT ----------------

if st.session_state.chunks:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about the document")

    if prompt:
        if not api_key:
            st.error("Please enter your Bytez API key.")
        else:
            st.session_state.history.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                srcs = retrieve(
                    prompt,
                    st.session_state.chunks,
                    st.session_state.vecs,
                    st.session_state.vec_fn,
                    top_k,
                )
                answer = ask(
                    prompt,
                    srcs,
                    st.session_state.history[:-1],
                    api_key,
                    model_id,
                )

            st.session_state.history.append({"role": "assistant", "content": answer})
            st.rerun()

else:
    st.info(" Upload a document or enter a URL to start chatting.")
              
