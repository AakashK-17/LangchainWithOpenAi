# LangChain with OpenAI

A hands-on learning project covering the core components of a Retrieval-Augmented Generation (RAG) pipeline using LangChain. The project is organized as a series of Jupyter notebooks, each focused on a specific stage of the pipeline: data ingestion, text transformation, embeddings, and vector stores.

## Project Structure

```
LangchainWithOpenAi/
├── GettingStarted.ipynb
├── requirements.txt
├── Data Ingestion/
│   ├── DataIngestion.ipynb
│   └── Attention.pdf
├── DataTransformation/
│   ├── CharacterTextSplitter.ipynb
│   ├── HTMLTextSplitter.ipynb
│   ├── RecursiveCharacterTextSplitter.ipynb
│   ├── RecursiveJsonSplitter.ipynb
│   ├── Attention.pdf
│   └── speech.txt
├── Embeddings/
│   ├── embeddings.ipynb
│   ├── huggingface.ipynb
│   └── ollamaembedding.ipynb
└── VectorStore/
    ├── chroma.ipynb
    ├── Faiss.ipynb
    └── speech.txt
```

## Modules

### 1. Data Ingestion

Located in `Data Ingestion/DataIngestion.ipynb`.

Demonstrates how to load documents from various sources using LangChain document loaders.

| Loader | Source Type |
|---|---|
| TextLoader | Plain text files |
| PyPDFLoader | PDF documents |
| WebBaseLoader | Web pages (via BeautifulSoup) |
| ArxivLoader | Academic papers from arXiv |
| WikipediaLoader | Wikipedia articles |

All loaders return LangChain `Document` objects containing `page_content` and a `metadata` dictionary with source-specific fields such as page number, author, or URL.

---

### 2. Data Transformation

Located in `DataTransformation/`. Covers four text splitters that prepare raw documents for embedding.

**CharacterTextSplitter** (`CharacterTextSplitter.ipynb`)
Splits text on a configurable separator (default `\n\n`) with `chunk_size` and `chunk_overlap` parameters. Works on both raw strings and `Document` objects.

**HTMLHeaderTextSplitter** (`HTMLTextSplitter.ipynb`)
Splits HTML documents by header hierarchy (h1, h2, h3). Stores the header context in each chunk's metadata, preserving the document structure alongside the content.

**RecursiveCharacterTextSplitter** (`RecursiveCharacterTextSplitter.ipynb`)
Splits text using a hierarchy of separators (paragraphs, sentences, then characters) to produce semantically coherent chunks. Preferred over `CharacterTextSplitter` for most use cases.

**RecursiveJsonSplitter** (`RecursiveJsonSplitter.ipynb`)
Splits JSON objects while keeping each chunk as valid JSON. Useful for structured API responses or configuration data. Demonstrated using the LangSmith OpenAPI specification.

---

### 3. Embeddings

Located in `Embeddings/`. Covers three different providers for converting text to vector representations.

**OpenAI Embeddings** (`embeddings.ipynb`)
Uses `langchain_openai.OpenAIEmbeddings` with the `text-embedding-3-small` model. Requires an `OPENAI_API_KEY` environment variable. Produces 1536-dimensional vectors by default, which can be reduced (e.g., to 1024) via the `dimensions` parameter. Also demonstrates the full pipeline from document loading through splitting to storage in a Chroma vector store.

**HuggingFace Embeddings** (`huggingface.ipynb`)
Uses `langchain_huggingface.HuggingFaceEmbeddings` with the `all-MiniLM-L6-v2` sentence transformer model. Runs entirely locally using `sentence_transformers`. No API key is required. Produces 384-dimensional vectors.

**Ollama Embeddings** (`ollamaembedding.ipynb`)
Uses `langchain_community.embeddings.OllamaEmbeddings` with a locally running Ollama server (default `http://localhost:11434`). Demonstrated with the `gemma:2b` model, producing 2048-dimensional vectors. Supports custom `embed_instruction` and `query_instruction` parameters. Fully offline.

---

### 4. Vector Stores

Located in `VectorStore/`. Covers two vector databases with similarity search and disk persistence.

**Chroma** (`chroma.ipynb`)
Uses `langchain_chroma.Chroma` backed by OllamaEmbeddings. Demonstrates:
- Creating a vector store from documents with `Chroma.from_documents()`
- `similarity_search()` for basic semantic search
- `similarity_search_with_score()` for results with relevance scores
- Persisting the index to disk via `persist_directory`
- Reloading a persisted database in a new session
- Wrapping the store as a standard LangChain `Retriever` with `.as_retriever()`

**FAISS** (`Faiss.ipynb`)
Uses `langchain_community.vectorstores.FAISS` backed by HuggingFaceEmbeddings. Demonstrates:
- Creating an index with `FAISS.from_documents()`
- `similarity_search()` and `similarity_search_with_score()`
- `similarity_search_with_score_by_vector()` for direct vector queries
- Saving and loading the index locally via `save_local()` and `load_local()`
- Wrapping as a LangChain `Retriever`

---

## Setup

### Prerequisites

- Python 3.9 or higher
- Jupyter Notebook or JupyterLab
- Ollama installed and running locally (required for Ollama notebooks)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/AakashK-17/LangchainWithOpenAi.git
cd LangchainWithOpenAi
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root for API keys:

```
OPENAI_API_KEY=your_openai_api_key
```

Only the `Embeddings/embeddings.ipynb` notebook requires an OpenAI API key. All other notebooks use local models (HuggingFace or Ollama) and work without any API key.

### Ollama Setup

The Ollama notebooks require a running Ollama server with the `gemma:2b` model pulled:

```bash
ollama pull gemma:2b
ollama serve
```

---

## Dependencies

| Package | Purpose |
|---|---|
| langchain | Core LangChain framework |
| langchain-community | Community integrations (loaders, Ollama, FAISS) |
| langchain-text-splitters | Text chunking utilities |
| langchain-openai | OpenAI embeddings and models |
| langchain-huggingface | HuggingFace embeddings |
| langchain-chroma | Chroma vector store integration |
| chromadb | Chroma vector database |
| faiss-cpu | FAISS vector similarity search |
| sentence-transformers | Local HuggingFace embedding models |
| pypdf | PDF document loading |
| pymupdf | PDF handling |
| bs4 | Web page parsing (BeautifulSoup) |
| arxiv | arXiv paper loader |
| wikipedia | Wikipedia article loader |
| python-dotenv | Environment variable management |
| ipykernel | Jupyter notebook kernel |

---

## RAG Pipeline Overview

The modules in this project map directly to the stages of a standard RAG pipeline:

```
Data Sources --> Document Loaders --> Text Splitters --> Embedding Models --> Vector Stores
(PDF, Web,       (DataIngestion)      (DataTransforma-   (Embeddings)        (VectorStore)
 arXiv, Wiki)                          tion)
```

Each module can be used independently or chained together. The `Embeddings/embeddings.ipynb` notebook demonstrates the full end-to-end pipeline in a single notebook.
