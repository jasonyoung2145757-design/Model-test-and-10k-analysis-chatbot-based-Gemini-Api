import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import csv
import json
import hashlib
import shutil
from io import StringIO
from pathlib import Path
from datetime import datetime

import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings


# =========================================================
# CONFIGURATION
# =========================================================

EMBED_MODEL = "qwen3-embedding:0.6b"

MODELS = [
    "qwen3.5:4b",
    "qwen3.5:9b",
    "deepseek-r1:14b",
]

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
RETRIEVER_K = 3
TEMPERATURE = 0
MAX_TOKENS = 2048

INDEX_DIR = Path("./faiss_index_locked")
TMP_PDF_DIR = Path("./_tmp_pdfs")
INDEX_DIR.mkdir(exist_ok=True)
TMP_PDF_DIR.mkdir(exist_ok=True)

PROMPT_TEMPLATE = """You are a professional financial analyst.

Answer the user's question using ONLY the provided filing context.

Rules:
1. If the answer is not explicitly supported by the context, reply exactly: Not found in filings.
2. Do not use outside knowledge.
3. Prefer exact figures and accounting terms from the context.
4. Keep the answer concise.
5. If the context contains the answer, provide the answer directly.
6. Do not output blank.
7. Do not explain your reasoning.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# =========================================================
# UTILITIES
# =========================================================

def build_faiss_with_batched_embeddings(chunks, embeddings, batch_size=16):
    print(f"[DEBUG] Starting batched embeddings for {len(chunks)} chunks...")

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    all_vectors = []
    total = len(texts)

    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_texts = texts[start_idx:end_idx]

        print(f"[DEBUG] Embedding batch {start_idx}-{end_idx - 1} / {total - 1}")

        t0 = time.time()
        batch_vectors = embeddings.embed_documents(batch_texts)
        elapsed = time.time() - t0

        print(
            f"[DEBUG] Batch completed: {start_idx}-{end_idx - 1} | "
            f"size={len(batch_vectors)} | elapsed={elapsed:.2f}s"
        )

        all_vectors.extend(batch_vectors)

    print("[DEBUG] All embedding batches completed. Building FAISS index from vectors...")

    vs = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, all_vectors)),
        embedding=embeddings,
        metadatas=metadatas
    )

    print("[DEBUG] FAISS index built from batched embeddings.")
    return vs


def make_file_hash(uploaded_files):
    h = hashlib.md5()
    for f in sorted(uploaded_files, key=lambda x: x.name):
        data = f.getvalue()
        h.update(f.name.encode("utf-8"))
        h.update(data)
    return h.hexdigest()


def make_index_name(uploaded_files):
    file_hash = make_file_hash(uploaded_files)
    raw = f"{EMBED_MODEL}|{file_hash}|{CHUNK_SIZE}|{CHUNK_OVERLAP}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def persist_uploaded_files(uploaded_files):
    paths = []

    for f in uploaded_files:
        file_bytes = f.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()[:8]
        safe_name = f"{file_hash}_{Path(f.name).name}"
        p = TMP_PDF_DIR / safe_name

        if not p.exists():
            p.write_bytes(file_bytes)

        paths.append(str(p))

    return paths


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)


def get_llm(model_name):
    return ChatOllama(
        model=model_name,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
    )


def load_or_build_index(pdf_paths, embeddings, index_name):
    path = INDEX_DIR / index_name

    if path.exists():
        print(f"[DEBUG] Loading existing FAISS index: {path}")
        return FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True
        )

    docs = []
    for p in pdf_paths:
        print(f"[DEBUG] Loading PDF: {p}")
        loader = PDFPlumberLoader(p)
        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.metadata["source"] = Path(p).name

        docs.extend(loaded_docs)

    print(f"[DEBUG] Total loaded pages/docs: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(docs)
    print(f"[DEBUG] Total chunks created: {len(chunks)}")

    print("[DEBUG] Starting embeddings + FAISS index build...")
    t0 = time.time()

    vs = build_faiss_with_batched_embeddings(
        chunks=chunks,
        embeddings=embeddings,
        batch_size=16
    )

    print(f"[DEBUG] FAISS index build completed in {time.time() - t0:.2f}s")

    print(f"[DEBUG] Saving FAISS index to: {path}")
    vs.save_local(str(path))
    print("[DEBUG] FAISS index saved.")

    return vs


def retrieve_context(vector_db, question):
    print(f"[DEBUG] Retrieving context for question: {question}")

    retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_K})
    docs = retriever.invoke(question)

    print(f"[DEBUG] Retrieved docs count: {len(docs)}")
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "N/A")
        preview = d.page_content[:180].replace("\n", " ")
        print(f"[DEBUG] Retrieved #{i} | source={source} | page={page} | preview={preview}")

    return docs


def format_context(docs):
    if not docs:
        return ""

    blocks = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        text = doc.page_content.strip()

        blocks.append(
            f"[Chunk {i+1} | {source} | Page {page}]\n{text}"
        )

    return "\n\n".join(blocks)


def format_sources(docs):
    refs = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        refs.append(f"{source}#Page{page}")

    seen = set()
    deduped = []

    for r in refs:
        if r not in seen:
            seen.add(r)
            deduped.append(r)

    return "; ".join(deduped)


def extract_answer_from_result(result):
    """
    More robust extraction for different ChatOllama/LangChain return shapes.
    """
    print(f"[DEBUG] Raw result type: {type(result)}")
    print(f"[DEBUG] Raw result repr: {repr(result)}")

    answer = ""

    # Common case: AIMessage with .content
    if hasattr(result, "content"):
        content = result.content
        print(f"[DEBUG] result.content type: {type(content)}")
        print(f"[DEBUG] result.content repr: {repr(content)}")

        if isinstance(content, str):
            answer = content.strip()

        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if "text" in item and item["text"]:
                        parts.append(str(item["text"]))
                    elif "content" in item and item["content"]:
                        parts.append(str(item["content"]))
            answer = "\n".join(parts).strip()

    # Fallback: try useful metadata fields
    if not answer and hasattr(result, "additional_kwargs"):
        akw = result.additional_kwargs
        print(f"[DEBUG] additional_kwargs: {repr(akw)}")

        if isinstance(akw, dict):
            for key in ["response", "text", "output", "answer"]:
                value = akw.get(key)
                if isinstance(value, str) and value.strip():
                    answer = value.strip()
                    break

    if not answer and hasattr(result, "response_metadata"):
        rmeta = result.response_metadata
        print(f"[DEBUG] response_metadata: {repr(rmeta)}")

        if isinstance(rmeta, dict):
            for key in ["response", "text", "output", "answer"]:
                value = rmeta.get(key)
                if isinstance(value, str) and value.strip():
                    answer = value.strip()
                    break

    # Last fallback
    if not answer:
        raw = str(result).strip()
        print("[DEBUG] Fallback to str(result)")
        if raw and raw != "content=''":
            answer = raw

    return clean_answer(answer)


def clean_answer(answer):
    if answer is None:
        return ""

    if not isinstance(answer, str):
        answer = str(answer)

    answer = answer.strip()

    # remove empty json-ish junk or blank-like output
    if answer in ["", "{}", "[]", "None", "null"]:
        return ""

    return answer


def run_model(question, docs, model_name):
    print(f"[DEBUG] Preparing model: {model_name}")

    llm = get_llm(model_name)
    context = format_context(docs)

    if not context.strip():
        return {
            "model": model_name,
            "answer": "Not found in filings.",
            "latency": 0.0,
            "sources": docs,
            "status": "success",
            "error": ""
        }

    prompt = PROMPT.format(
        context=context,
        question=question
    )

    print(f"[DEBUG] Invoking model: {model_name}")
    start = time.time()
    result = llm.invoke(prompt)
    latency = time.time() - start
    print(f"[DEBUG] Model completed: {model_name} | Latency: {latency:.2f}s")

    answer = extract_answer_from_result(result)

    if not answer:
        print(f"[DEBUG] Empty extracted answer for {model_name}, forcing fallback text.")
        answer = "Not found in filings."

    return {
        "model": model_name,
        "answer": answer,
        "latency": latency,
        "sources": docs,
        "status": "success",
        "error": ""
    }


def ensure_session_state():
    if "benchmark_rows" not in st.session_state:
        st.session_state.benchmark_rows = []


def append_benchmark_rows(question, results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for r in results:
        st.session_state.benchmark_rows.append({
            "time": timestamp,
            "question": question,
            "model": r["model"],
            "status": r.get("status", "unknown"),
            "latency": round(r["latency"], 2) if r.get("latency") is not None else None,
            "answer": r.get("answer", ""),
            "error": r.get("error", ""),
            "sources": format_sources(r.get("sources", [])),
        })


def build_csv(rows):
    csv_buffer = StringIO()

    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "time",
            "question",
            "model",
            "status",
            "latency",
            "answer",
            "error",
            "sources",
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

    return csv_buffer.getvalue()


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title="Local RAG Benchmark",
    layout="wide"
)

ensure_session_state()

st.title("Financial Filing RAG Benchmark")

st.caption(
    f"Embedding: {EMBED_MODEL} | "
    f"Chunk={CHUNK_SIZE} | "
    f"Overlap={CHUNK_OVERLAP} | "
    f"TopK={RETRIEVER_K} | "
    f"MaxTokens={MAX_TOKENS}"
)

with st.sidebar:
    st.header("Experiment Settings")

    st.write("Embedding Model")
    st.code(EMBED_MODEL)

    st.write("Models Under Test")
    for m in MODELS:
        st.write(m)

    if st.button("Clear Vector Index"):
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
        INDEX_DIR.mkdir(exist_ok=True)
        st.success("Vector index cleared.")

    if st.button("Clear Benchmark History"):
        st.session_state.benchmark_rows = []
        st.success("Benchmark history cleared.")

    uploaded_files = st.file_uploader(
        "Upload Filing PDFs",
        type="pdf",
        accept_multiple_files=True
    )

question = st.chat_input("Ask a financial filing question...")

if question:
    if not uploaded_files:
        st.error("Please upload at least one PDF filing.")
    else:
        pdf_paths = persist_uploaded_files(uploaded_files)
        index_name = make_index_name(uploaded_files)
        embeddings = get_embeddings()

        try:
            with st.spinner("Building / loading vector index..."):
                vector_db = load_or_build_index(
                    pdf_paths,
                    embeddings,
                    index_name
                )

            with st.spinner("Retrieving relevant context..."):
                docs = retrieve_context(
                    vector_db,
                    question
                )

            if not docs:
                st.warning("No relevant chunks were retrieved.")
            else:
                st.subheader("Retrieved Context")

                with st.expander("View retrieved chunks", expanded=False):
                    for i, d in enumerate(docs, start=1):
                        source = d.metadata.get("source", "unknown")
                        page = d.metadata.get("page", "N/A")
                        st.markdown(f"**Chunk {i} | {source} | Page {page}**")
                        preview = d.page_content[:1200]
                        if len(d.page_content) > 1200:
                            preview += "..."
                        st.write(preview)

            results = []

            st.subheader("Model Benchmark")

            for model in MODELS:
                st.markdown(f"### {model}")

                status_box = st.empty()
                answer_box = st.empty()
                info_box = st.empty()

                print(f"[DEBUG] About to run model: {model}")
                status_box.info(f"Running {model}...")

                try:
                    r = run_model(question, docs, model)
                    results.append(r)

                    answer_text = clean_answer(r["answer"])
                    if not answer_text:
                        answer_text = "Not found in filings."

                    answer_box.markdown(answer_text)
                    info_box.success(
                        f"Done | Latency: {r['latency']:.2f}s | Sources: {format_sources(r['sources'])}"
                    )
                    status_box.empty()

                except Exception as e:
                    error_text = repr(e)
                    print(f"[DEBUG] Model failed: {model} | Error: {error_text}")

                    failed_result = {
                        "model": model,
                        "answer": "",
                        "latency": None,
                        "sources": docs,
                        "status": "failed",
                        "error": error_text
                    }
                    results.append(failed_result)

                    answer_box.error(f"{model} failed.\n\nError: {error_text}")
                    info_box.warning("Execution stopped for this model, continuing to the next one.")
                    status_box.empty()

                st.divider()

            append_benchmark_rows(question, results)

            st.subheader("Run Summary")

            summary_rows = []
            for r in results:
                summary_rows.append({
                    "Model": r["model"],
                    "Status": r.get("status", ""),
                    "Latency (s)": round(r["latency"], 2) if r.get("latency") is not None else None,
                    "Answer": clean_answer(r.get("answer", "")),
                    "Sources": format_sources(r.get("sources", [])),
                    "Error": r.get("error", "")
                })

            st.dataframe(summary_rows, use_container_width=True)

        except Exception as e:
            st.error(f"Application error: {repr(e)}")
            print(f"[DEBUG] Application-level failure: {repr(e)}")


# =========================================================
# BENCHMARK TABLE
# =========================================================

st.divider()
st.subheader("Benchmark History")

if st.session_state.benchmark_rows:
    st.dataframe(st.session_state.benchmark_rows, use_container_width=True)

    csv_data = build_csv(st.session_state.benchmark_rows)

    st.download_button(
        "Download CSV",
        csv_data,
        "benchmark_results.csv",
        mime="text/csv"
    )
else:
    st.write("No benchmark runs yet.")