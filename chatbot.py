"""
RAG Chatbot for 10-K Financial Document Analysis
=================================================
BU.520.710 - AI Essentials for Business - Final Project
JHU Carey Business School

Usage:
    streamlit run chatbot.py
"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# ─── LangChain imports (compatible with v1.2.x) ────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# =====================================================================
# CONFIGURATION — Change EMBEDDING_TYPE to compare different embeddings
# =====================================================================

# ┌─────────────────────────────────────────────────────────────────┐
# │  EMBEDDING OPTIONS — just change this one line to switch:       │
# │                                                                 │
# │  "gemini"       → Google Gemini (free, fast, cloud)             │
# │  "openai"       → OpenAI (paid, high quality, cloud)            │
# │  "ollama"       → Ollama nomic-embed-text (free, local, slow)   │
# │  "huggingface"  → HuggingFace all-MiniLM-L6-v2 (free, local)   │
# └─────────────────────────────────────────────────────────────────┘
EMBEDDING_TYPE = "ollama"

#gemini

#ollama

#EMBEDDING_TYPE = "openai"
#EMBEDDING_MODEL = "text-embedding-3-small"

# LLM via OpenRouter (this stays the same regardless of embedding choice)
LLM_MODEL = "google/gemini-2.0-flash-001"

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 5
PDF_FOLDER = "./10k_files"

# Vector store cache — separate folder per embedding type so they don't overwrite
VECTORSTORE_PATH = f"./vectorstore_{EMBEDDING_TYPE}"


# =====================================================================
# SYSTEM PROMPT
# =====================================================================

SYSTEM_PROMPT = """You are a senior financial analyst assistant specializing in 
analyzing 10-K annual reports filed with the SEC. You have access to the latest 
10-K filings from Alphabet (Google), Amazon, and Microsoft.

Your guidelines:
1. Answer questions ONLY based on the provided context from the 10-K documents.
2. When citing financial figures, be precise with exact numbers, units, and period.
3. If comparing companies, clearly label which company each data point belongs to.
4. If the context does not contain enough information, say so honestly.
5. When discussing risks, reference the specific risk factors from the filings.
6. For revenue questions, distinguish between segments (e.g., AWS vs retail).
7. Always specify which fiscal year the data comes from.

If you're unsure, say "Based on the provided 10-K excerpts, I cannot find specific 
information about that" rather than making something up.

CONTEXT FROM 10-K FILINGS:
{context}"""


# =====================================================================
# EMBEDDING MODEL FACTORY — supports 4 providers
# =====================================================================

def _get_embeddings():
    """
    Return the configured embedding model.
    Switch between providers by changing EMBEDDING_TYPE at the top.
    """

    if EMBEDDING_TYPE == "gemini":
        # Google Gemini — free tier, cloud-based, fast
        # Model: gemini-embedding-001 (latest as of March 2026)
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    elif EMBEDDING_TYPE == "openai":
        # OpenAI — paid, cloud-based, high quality
        # Requires OPENAI_API_KEY in .env (NOT OpenRouter — actual OpenAI key)
        # Models: text-embedding-3-small (cheap), text-embedding-3-large (better)
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif EMBEDDING_TYPE == "ollama":
        # Ollama — free, runs locally, no API key needed
        # Requires: ollama serve running + ollama pull nomic-embed-text
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model="nomic-embed-text",
        )

    elif EMBEDDING_TYPE == "huggingface":
        # HuggingFace Sentence Transformers — free, local, no API key
        # Downloads model on first run (~90MB), then cached locally
        # all-MiniLM-L6-v2 is small, fast, and surprisingly good
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
        )

    else:
        raise ValueError(f"Unknown embedding type: {EMBEDDING_TYPE}")


# =====================================================================
# CORE FUNCTIONS
# =====================================================================

@st.cache_resource(show_spinner="Loading and embedding 10-K documents...")
def build_vector_store():
    """Load PDFs -> Chunk -> Embed -> Store in FAISS."""

    # Check for cached vector store on disk
    if Path(VECTORSTORE_PATH).exists():
        embeddings = _get_embeddings()
        return FAISS.load_local(
            VECTORSTORE_PATH, embeddings,
            allow_dangerous_deserialization=True
        )

    # Load all PDFs
    pdf_path = Path(PDF_FOLDER)
    if not pdf_path.exists():
        st.error(f"PDF folder '{PDF_FOLDER}' not found!")
        st.stop()

    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        st.error(f"No PDF files found in '{PDF_FOLDER}'!")
        st.stop()

    all_docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        company = _identify_company(pdf_file.name)
        for doc in docs:
            doc.metadata["company"] = company
            doc.metadata["source_file"] = pdf_file.name
        all_docs.extend(docs)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    # Create embeddings and vector store
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    return vectorstore


def _get_llm():
    """Return the LLM via OpenRouter."""
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=1500,
    )


def _identify_company(filename: str) -> str:
    """Identify company from PDF filename."""
    fname = filename.lower()
    if "google" in fname or "alphabet" in fname or "goog" in fname:
        return "Alphabet (Google)"
    elif "amazon" in fname or "amzn" in fname:
        return "Amazon"
    elif "microsoft" in fname or "msft" in fname:
        return "Microsoft"
    return "Unknown"


def ask_question(vectorstore, question: str, chat_history: list) -> dict:
    """
    Simple RAG: retrieve relevant chunks, stuff into prompt, call LLM.
    """
    # Step 1: Retrieve relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    docs = retriever.invoke(question)

    # Step 2: Build context string from retrieved chunks
    context_parts = []
    for doc in docs:
        company = doc.metadata.get("company", "Unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[{company} - Page {page}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Include recent chat history for follow-up questions
    history_section = ""
    if chat_history:
        recent = chat_history[-6:]
        history_lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {msg['content'][:300]}")
        history_section = "Previous conversation:\n" + "\n".join(history_lines) + "\n\n"

    # Step 4: Build prompt and call LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{history_section}Question: {question}"),
    ])

    llm = _get_llm()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "history_section": history_section,
        "question": question,
    })

    # Step 5: Return answer + source info
    sources = []
    for doc in docs:
        sources.append({
            "company": doc.metadata.get("company", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "preview": doc.page_content[:200] + "...",
        })

    return {"answer": answer, "sources": sources}


# =====================================================================
# STREAMLIT UI
# =====================================================================

def main():
    st.set_page_config(
        page_title="10-K Financial Analyst",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 10-K Financial Analyst Chatbot")
    st.caption("Analyze Alphabet, Amazon, and Microsoft 10-K filings using RAG")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown(f"**Embedding:** {EMBEDDING_TYPE}")
        st.markdown(f"**LLM:** {LLM_MODEL}")
        st.markdown(f"**Chunk Size:** {CHUNK_SIZE} | **Overlap:** {CHUNK_OVERLAP}")
        st.markdown(f"**Top-K Retrieval:** {RETRIEVAL_K}")

        st.divider()

        st.header("📝 Sample Questions")
        sample_questions = [
            "How much cash does Amazon have at the end of 2024?",
            "What is the main revenue source for each company?",
            "Do these companies mention cloud service risks in China or India?",
            "Compared to 2023, did Amazon's liquidity increase or decrease?",
            "What are the main businesses Amazon operates in?",
            "Compare the cloud computing revenue of all three companies.",
            "What risk factors does Microsoft mention about AI?",
        ]
        for q in sample_questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state["pending_question"] = q

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("🔄 Rebuild Vector Store", use_container_width=True):
            import shutil
            if Path(VECTORSTORE_PATH).exists():
                shutil.rmtree(VECTORSTORE_PATH)
            st.cache_resource.clear()
            st.rerun()

    # --- Check API keys ---
    missing = []
    if EMBEDDING_TYPE == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    if EMBEDDING_TYPE == "openai" and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY")
    # ollama and huggingface need no API keys

    if missing:
        st.error(f"Missing API keys: {', '.join(missing)}")
        st.code('GOOGLE_API_KEY=your-key\nOPENROUTER_API_KEY=your-key', language="bash")
        st.stop()

    # --- Build vector store (cached) ---
    vectorstore = build_vector_store()

    # --- Chat history ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"- **{src['company']}** (Page {src['page']}): "
                            f"{src['preview']}"
                        )

    # --- Handle input ---
    pending = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask about the 10-K filings...") or pending

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing 10-K filings..."):
                try:
                    result = ask_question(
                        vectorstore, user_input, st.session_state.messages
                    )
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("📄 Sources"):
                            for src in sources:
                                st.markdown(
                                    f"- **{src['company']}** (Page {src['page']}): "
                                    f"{src['preview']}"
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


if __name__ == "__main__":
    main()
