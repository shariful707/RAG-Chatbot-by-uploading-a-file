# 🤖 Dynamic Conversational RAG Chatbot (Persistent)

A **persistent, document-aware chatbot** built with **Streamlit**, **FAISS**, and **Llama 3.1:latest** via **Ollama API**. This chatbot allows you to upload PDFs, automatically create embeddings, and answer user queries based on **retrieved context** from your documents.

---

## **Features**

- **RAG (Retrieval-Augmented Generation)**: Only retrieves and injects the most relevant chunks of your uploaded documents for context-aware answers.
- **Persistent Knowledge Base**: FAISS index and chat history are saved locally (`faiss_data` folder) for continuity between sessions.
- **PDF Support**: Upload PDF documents; text is automatically extracted, chunked, and embedded.
- **Citation Tracking**: LLM responses include citations pointing to the source document(s) used.
- **General Conversation**: Works even without uploaded documents.
- **Clear State Button**: Reset all uploaded documents and chat history easily.

---

## **Tech Stack**

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- **LLM**: [Llama 3.1:latest](https://ollama.com/) via Ollama API
- **PDF Parsing**: [pypdf](https://pypi.org/project/pypdf/)

---

## **Installation & Setup**

1. **Clone the repository**

```bash
git clone https://github.com/shariful707/RAG-Chatbot-by-uploading-a-file.git
cd RAG-Chatbot-by-uploading-a-file

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt


streamlit
requests
faiss-cpu
numpy
sentence-transformers
pypdf

ollama serve
streamlit run rag.py


faiss_data/               # Persistent folder storing FAISS index, metadata, and chat history
├── faiss_index.bin        # FAISS vector index
├── rag_chunks.json        # Chunk metadata
├── chat_history.json      # Persistent chat history

app.py                     # Main Streamlit application
requirements.txt           # Python dependencies
README.md                  # Project documentation
