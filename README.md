# RAG Chatbot for 10-K Financial Document Analysis

**BU.520.710 – AI Essentials for Business · Final Project**  
Johns Hopkins University, Carey Business School

This project builds a Retrieval-Augmented Generation (RAG) system that lets you chat with SEC 10-K annual filings from **Alphabet (Google)**, **Amazon**, and **Microsoft**. It includes a conversational chatbot that supports multiple embedding providers, and a local LLM benchmark tool for comparing model performance.

---

# 1. Prepare Your Tools

Feel free to skip any steps if you have already installed the tools.

## 1.1 Install Visual Studio Code (VSCode)

1. Go to the [VSCode download page](https://code.visualstudio.com/download)
2. Download the appropriate version for your operating system (Windows, macOS, or Linux)
3. Follow the installation instructions for your platform
4. Launch VSCode after installation

## 1.2 Download and Open This Repository

1. Download this repository by clicking the green **Code** button on the GitHub page and selecting **Download ZIP**
2. Extract the ZIP file to a location on your computer
3. In VSCode, go to **File > Open Folder** and select the extracted folder

## 1.3 Opening the Terminal in VSCode

1. Press `` Ctrl+` `` (Windows/Linux) or `` Cmd+` `` (macOS) to open the integrated terminal
2. Alternatively, go to **View > Terminal** from the menu bar

## 1.4 Install Miniconda

1. Go to the [Miniconda download page](https://www.anaconda.com/docs/getting-started/miniconda/install)
2. Download the appropriate installer for your operating system
3. Run the installer and follow the instructions
4. Verify the installation by opening a new terminal and typing:

```bash
conda --version
```

---

# 2. Install the Environment

Create a new conda environment named `chatbot` and install all required packages.

```bash
# Create and activate the environment
conda create -n chatbot python=3.11
conda activate chatbot

# Install packages via conda-forge
conda install -c conda-forge streamlit faiss-cpu pillow

# Install LangChain and provider integrations
pip install "langchain>=0.1.0"
pip install "langchain-community>=0.0.10"
pip install "langchain-text-splitters"
pip install "langchain-google-genai>=0.0.5"
pip install "langchain-openai>=0.0.2"
pip install "langchain-ollama"
pip install "langchain-huggingface"

# Install model SDKs and utilities
pip install "openai>=1.3.0"
pip install "google-generativeai>=0.3.0"
pip install "sentence-transformers"
pip install "pypdf>=3.15.1"
pip install "pdfplumber"
pip install "python-dotenv"
```

Confirm the environment is working:

```bash
conda activate chatbot
which python  # expected: /Users/your_username/miniconda3/envs/chatbot/bin/python
```

## 2.1 Add Your 10-K PDF Files

Place the 10-K PDF files you want to analyze inside the `10k_files/` folder. The app identifies companies automatically based on the filename:

| Company | Filename should contain |
|---|---|
| Alphabet / Google | `google`, `alphabet`, or `goog` |
| Amazon | `amazon` or `amzn` |
| Microsoft | `microsoft` or `msft` |

Example:
```
10k_files/
├── alphabet_10k_2024.pdf
├── amazon_10k_2024.pdf
└── microsoft_10k_2024.pdf
```

---

# 3. Run the Chatbot

The main chatbot (`chatbot.py`) supports four embedding providers. You only need to set up the one you want to use.

## Option A — Ollama (Local, Free, No API Key)

Ollama lets you run embedding models locally with no internet connection or API key required.

### 3.1 Install Ollama

1. Visit [https://ollama.com/download](https://ollama.com/download)
2. Download and install the version for your operating system
3. After installation, Ollama will run as a background service
4. Verify the installation:

```bash
ollama --version  # expected: a version number like 0.5.x
```

### 3.2 Pull the Required Models

Pull the embedding model used by the chatbot:

```bash
ollama pull nomic-embed-text
```

You can verify the embedding model is working correctly by running the test script:

```bash
python test_embed.py
# expected output: num_vectors = 3, dim = 768, elapsed = ...
```

### 3.3 Configure and Run

In `chatbot.py`, make sure the embedding type is set to Ollama (this is the default):

```python
EMBEDDING_TYPE = "ollama"
```

Launch the chatbot:

```bash
streamlit run chatbot.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. The first run will build and cache the FAISS vector index — this may take a few minutes depending on the number of PDFs. Subsequent runs load the index from disk instantly.

---

## Option B — Google Gemini (Cloud, Free Tier)

### 3.1 Get a Gemini API Key

Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey) and create a free API key.

### 3.2 Set Up the API Key

Create a `.env` file in the project root folder:

```env
GOOGLE_API_KEY=your-google-api-key-here
OPENROUTER_API_KEY=your-openrouter-api-key-here
```

### 3.3 Configure and Run

In `chatbot.py`, set the embedding type to Gemini:

```python
EMBEDDING_TYPE = "gemini"
```

Launch the chatbot:

```bash
streamlit run chatbot.py
```

---

## Option C — OpenAI (Cloud, Paid)

### 3.1 Get an OpenAI API Key

Sign in to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) and create an API key.

### 3.2 Set Up the API Key

Add it to your `.env` file:

```env
OPENAI_API_KEY=your-openai-api-key-here
OPENROUTER_API_KEY=your-openrouter-api-key-here
```

### 3.3 Configure and Run

In `chatbot.py`, set the embedding type to OpenAI:

```python
EMBEDDING_TYPE = "openai"
```

Launch the chatbot:

```bash
streamlit run chatbot.py
```

---

## Option D — HuggingFace (Local, Free, No API Key)

This option downloads the `all-MiniLM-L6-v2` model (~90 MB) on first run and caches it locally. No API key needed.

In `chatbot.py`, set the embedding type to HuggingFace:

```python
EMBEDDING_TYPE = "huggingface"
```

Launch the chatbot:

```bash
streamlit run chatbot.py
```

---

## LLM via OpenRouter

Regardless of which embedding provider you choose, the chatbot uses **OpenRouter** to call the LLM (`google/gemini-2.0-flash-001` by default).

Get a free OpenRouter API key at [https://openrouter.ai/keys](https://openrouter.ai/keys) and add it to your `.env` file:

```env
OPENROUTER_API_KEY=your-openrouter-api-key-here
```

---

# 4. Local LLM Benchmark Tool

In addition to the chatbot, this project includes a benchmark tool that runs the same financial question against multiple local Ollama models and compares their answers and response times.

## 4.1 Pull the Required Models

```bash
ollama pull qwen3-embedding:0.6b   # embedding model used by the benchmark
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
ollama pull deepseek-r1:14b
```

> **Note:** The 14B model requires at least 16 GB of RAM. If your machine has less memory, you can remove it from the `MODELS` list at the top of the benchmark script.

## 4.2 Run the Benchmark

Three benchmark scripts are included, each testing a different chunking and retrieval configuration:

| Script | Chunk Size | Overlap | Top-K | Max Tokens |
|---|---|---|---|---|
| `rag_benchmark_local.py` | 600 | 120 | 3 | 2048 |
| `rag_benchmark_local2.py` | 400 | 80 | 2 | 256 |
| `rag_benchmark_local3.py` | 400 | 80 | 2 | 1024 |

Run any of them with:

```bash
streamlit run rag_benchmark_local.py
```

Upload one or more PDF filings via the sidebar, type your question, and the tool will display each model's answer alongside its response latency. All results can be exported as a CSV file.

---

# 5. Project File Overview

```
.
├── chatbot.py                  # Main RAG chatbot (multi-provider embeddings)
├── rag_benchmark_local.py      # Benchmark: chunk=600, topK=3, maxTokens=2048
├── rag_benchmark_local2.py     # Benchmark: chunk=400, topK=2, maxTokens=256
├── rag_benchmark_local3.py     # Benchmark: chunk=400, topK=2, maxTokens=1024
├── test_embed.py               # Embedding smoke test for Ollama
├── 10k_files/                  # Place your 10-K PDFs here
├── vectorstore_ollama/         # Auto-generated FAISS index (created on first run)
├── vectorstore_gemini/         # Auto-generated FAISS index (created on first run)
├── vectorstore_openai/         # Auto-generated FAISS index (created on first run)
├── vectorstore_huggingface/    # Auto-generated FAISS index (created on first run)
└── .env                        # API keys — do NOT commit this file
```

> Make sure `.env` is listed in your `.gitignore` so your API keys are never pushed to GitHub.

---

# 6. Sample Questions

Once the chatbot is running, try questions like:

- How much cash does Amazon have at the end of 2024?
- What is the main revenue source for each company?
- Compare the cloud computing revenue of all three companies.
- What risk factors does Microsoft mention about AI?
- Compared to 2023, did Amazon's liquidity increase or decrease?
- Do these companies mention cloud service risks in China or India?
