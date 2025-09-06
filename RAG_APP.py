import streamlit as st
import tempfile
import hashlib
import time
import io
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

import ollama  # <-- NEW

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# CONFIG
# =========================
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3:latest"   # Ollama local model

# =========================
# UTILITIES
# =========================
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def extract_text_pdf_then_ocr(pdf_path: str) -> str:
    """Extract text from PDF and fallback to OCR if needed"""
    doc = fitz.open(pdf_path)
    native_texts = [page.get_text() for page in doc]
    joined = "\n".join(native_texts).strip()

    if len(joined) < 200:  # fallback to OCR
        ocr_pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_pages.append(pytesseract.image_to_string(img))
        joined = "\n".join(ocr_pages)

    return joined

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="📄 Resume Analyzer", layout="wide")

st.markdown(
    """
    <h1 style="
        text-align:center; font-weight:800; letter-spacing:0.3px; margin-bottom:6px;
        background: linear-gradient(90deg, #ff6a00, #ee0979, #2196f3, #00c853);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        📄 Resume Analyzer
    </h1>
    <p style="text-align:center; color:#666; margin-top:0;">
        <em>Upload your resume PDF and get AI-driven ATS insights ⚡</em>
    </p>
    <hr/>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("Reset session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.success("Session reset. Re-upload a document.")

# --------------------
# SESSION STATE
# --------------------
for key, default in [
    ("doc_hash", None),
    ("vectorstore", None),
    ("embedded", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------
# UPLOAD PDF
# --------------------
uploaded_file = st.file_uploader("📂 Upload Resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    current_hash = sha256_bytes(file_bytes)

    if st.session_state.doc_hash != current_hash:
        st.session_state.doc_hash = current_hash
        st.session_state.vectorstore = None
        st.session_state.embedded = False

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            pdf_path = tmp.name

        # Extract text
        t0 = time.time()
        with st.spinner("🧠 Extracting text..."):
            full_text = extract_text_pdf_then_ocr(pdf_path)
        t1 = time.time()
        st.success(f"✅ Text extracted in {t1 - t0:.2f}s")

        if not full_text.strip():
            st.error("❌ Could not extract any text from the PDF.")
            st.stop()

        # Chunk & embed
        t2 = time.time()
        with st.spinner("⚡ Creating embeddings..."):
            st.markdown("<div style='text-align:center; font-size:58px;'>🧠</div>", unsafe_allow_html=True)
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = splitter.split_text(full_text)
            embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            st.session_state.vectorstore = Chroma.from_texts(chunks, embedding=embedder)
            st.session_state.embedded = True
        t3 = time.time()
        st.success(f"✅ Embedded in {t3 - t2:.2f}s")

# --------------------
# Q&A / Resume Analysis
# --------------------
if st.session_state.embedded and st.session_state.vectorstore is not None:
    st.subheader("💬 Ask about this resume")
    question = st.text_input("E.g., Analyze this resume for ATS score, summary, and suggestions…")

    if question:
        thinking = st.empty()
        thinking.markdown("<div style='text-align:center; font-size:64px;'>🤔</div>", unsafe_allow_html=True)
        tqa0 = time.time()

        # Retrieve relevant chunks
        docs = st.session_state.vectorstore.similarity_search(question, k=4)
        context = "\n\n".join([d.page_content for d in docs]) if docs else "No context found in document."

        prompt = f"""
You are an expert ATS resume evaluator.

- Answer the user’s question **strictly and only** based on the context provided.
- Do not include extra details that were not asked.
- If the user asks for ATS Score → return only the numeric score.
- If the user asks for Summary → return only the summary.
- If the user asks for Suggestions/Improvements → return only those.
- If the user requests multiple items (e.g., "Name and ATS Score"),
  return **all requested items** clearly, in separate lines.
- If the user asks for a full Analysis → return summary, ATS score, strengths, weaknesses, and improvements.
- If the context does not contain enough information, say: "Not enough information in the resume."

Context:
{context}

User Question: {question}
"""

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            answer_text = response['message']['content']
        except Exception as e:
            answer_text = f"❌ Error generating analysis: {e}"

        tqa1 = time.time()
        thinking.empty()

        st.markdown(
            f"""
            <div style="background:#f4faff; padding:16px; border:1px solid #dbe9ff; border-radius:12px;">
                <b>Answer:</b><br>{answer_text}
            </div>
            <div style="text-align:right; color:#888; font-size:12px; margin-top:6px;">
                ⏱️ {tqa1 - tqa0:.2f}s
            </div>
            """,
            unsafe_allow_html=True,
        )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-style:italic; color:#777;'>"
    "“Turn feedback into action.”<br>"
    "“Powered by Ollama Resume Analyzer.”"
    "</p>",
    unsafe_allow_html=True,
)
