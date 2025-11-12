import streamlit as st
import requests
import json
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from io import BytesIO

# --- Configuration ---
OLLAMA_CHAT_ENDPOINT = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIR = "faiss_data" 
INDEX_FILE = os.path.join(PERSIST_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(PERSIST_DIR, "rag_chunks.json")
HISTORY_FILE = os.path.join(PERSIST_DIR, "chat_history.json") # NEW: File for conversation history

# --- Initial Setup ---
@st.cache_resource
def load_embedding_model():
    """Load the Sentence Transformer model once."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

embedding_model = load_embedding_model()

# Initialize session state for the chat history and the knowledge base
if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = {"index": None, "chunks": [], "filenames": []}

# --- Persistence Functions ---

def save_persistence_data():
    """Saves the FAISS index, chunk metadata, and conversation history to disk."""
    index = st.session_state.knowledge_base["index"]
    chunks = st.session_state.knowledge_base["chunks"]
    filenames = st.session_state.knowledge_base["filenames"]
    messages = st.session_state.messages # Get history

    # 1. Create directory if it doesn't exist
    os.makedirs(PERSIST_DIR, exist_ok=True)

    # 2. Save FAISS index and RAG metadata (only if an index exists)
    if index is not None:
        faiss.write_index(index, INDEX_FILE)
        metadata = {"chunks": chunks, "filenames": filenames}
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        
    # 3. Save Conversation History (always save, even if empty)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)
    
    st.sidebar.success(f"✅ State saved to '{PERSIST_DIR}'.")


def load_persistence_data():
    """Loads the FAISS index, chunk metadata, and conversation history from disk."""
    rag_loaded = False
    
    # 1. Load FAISS Index and RAG Metadata
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            st.session_state.knowledge_base["index"] = index
            st.session_state.knowledge_base["chunks"] = metadata["chunks"]
            st.session_state.knowledge_base["filenames"] = metadata["filenames"]
            rag_loaded = True
        except Exception as e:
            st.sidebar.warning(f"Error loading RAG data: {e}. Starting fresh RAG base.")

    # 2. Load Conversation History
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                st.session_state.messages = json.load(f)
            history_loaded = True
        except Exception as e:
            st.sidebar.warning(f"Error loading chat history: {e}. Starting fresh chat.")

    if rag_loaded:
        st.sidebar.success(f"💾 RAG index and history loaded from '{PERSIST_DIR}'.")
    elif history_loaded:
        st.sidebar.success(f"💾 Chat history loaded from '{PERSIST_DIR}'.")
    return rag_loaded or history_loaded

# NEW: Attempt to load the index and history automatically on first run
if st.session_state.knowledge_base["index"] is None or not st.session_state.messages:
    load_persistence_data()

# --- Core RAG Functions ---

def get_document_text(file):
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(BytesIO(file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=512, chunk_overlap=50):
    """Simple text chunking logic."""
    chunks = []
    current_start = 0
    while current_start < len(text):
        chunk = text[current_start:current_start + chunk_size]
        chunks.append(chunk)
        current_start += chunk_size - chunk_overlap
    return chunks

def build_vector_store(text_chunks, filename):
    """Generate embeddings, build FAISS index, and save to disk."""
    # 1. Embed chunks
    embeddings = embedding_model.encode(text_chunks, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype('float32')

    # 2. Build FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 3. Store in session state
    st.session_state.knowledge_base["index"] = index
    st.session_state.knowledge_base["chunks"] = text_chunks
    st.session_state.knowledge_base["filenames"] = [filename] * len(text_chunks)
    st.success(f"Successfully loaded and embedded '{filename}' into FAISS.")
    
    # 4. Save the index and metadata
    save_persistence_data()


def retrieve_context(query, top_k=3):
    """Search FAISS for relevant document chunks."""
    index = st.session_state.knowledge_base["index"]
    chunks = st.session_state.knowledge_base["chunks"]
    filenames = st.session_state.knowledge_base["filenames"]

    query_embedding = embedding_model.encode([query], convert_to_tensor=False).astype('float32')

    D, I = index.search(query_embedding, top_k)
    
    context = []
    sources = set()
    for i in I[0]:
        if i < len(chunks):
            context.append(chunks[i])
            sources.add(filenames[i])
            
    return "\n\n---\n\n".join(context), list(sources)


# --- LLM Communication and RAG Logic ---

def ollama_chat(prompt_messages, system_prompt="You are a helpful and friendly AI assistant."):
    """Communicate with the Llama 3.1 model via the Ollama API with memory."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt}
            ] + prompt_messages,
            "stream": True
        }
        
        response = requests.post(OLLAMA_CHAT_ENDPOINT, json=payload, stream=True)
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'message' in data and 'content' in data['message']:
                    content = data['message']['content']
                    full_response += content
                    yield content
                
                if data.get("done"):
                    break
        
        return full_response
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Ollama: {e}")
        st.error("Please ensure Ollama is running and the 'llama3.1' model is pulled.")
        return f"Error: Could not connect to the LLM. ({e})"


def handle_user_input(user_query):
    """Handles both normal conversation and RAG queries with context memory."""
    # 1. Add user message to conversation memory
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # ... RAG Context Retrieval Logic ...
    rag_context = ""
    sources = []
    if st.session_state.knowledge_base["index"] is not None:
        retrieved_context, sources = retrieve_context(user_query)
        rag_context = retrieved_context

    # ... System Prompt and RAG Injection Logic ...
    system_prompt = (
        "You are a dynamic and helpful AI assistant. You can have a normal conversation "
        "or answer questions based on an injected document. "
    )
    
    if rag_context:
        system_prompt += (
            "\n\n***CONTEXT FROM INJECTED DOCUMENT***:\n"
            f"{rag_context}\n\n"
            "***INSTRUCTION***: Answer the user's question *only* based on the CONTEXT provided above. "
            "If the answer is not in the context, politely state that you cannot answer from the provided document. "
            "Always include a citation at the end of your answer, mentioning the document name(s)."
        )
        
    chat_history_for_ollama = [
        {"role": msg["role"], "content": msg["content"]} 
        for msg in st.session_state.messages
    ]

    # ... Stream LLM Response ...
    full_llm_response = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        for chunk in ollama_chat(chat_history_for_ollama, system_prompt):
            full_llm_response += chunk
            message_placeholder.markdown(full_llm_response + "▌")
            
        message_placeholder.markdown(full_llm_response)

    # 2. Add final LLM response to conversation memory
    st.session_state.messages.append({"role": "assistant", "content": full_llm_response})
    
    # 3. NEW: Save the updated conversation history
    save_persistence_data()
    
    # 4. Display sources/citations dynamically
    if sources:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Source Document(s) Used:** `{', '.join(sources)}`")


# --- Streamlit UI (Modified) ---

st.title("🤖 Dynamic Conversational RAG Chatbot (Persistent)")
st.caption(f"LLM: {OLLAMA_MODEL} via Ollama | Embeddings: {EMBEDDING_MODEL_NAME} | Vector Store: FAISS")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Document (RAG)")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if st.button("Process Document") and uploaded_file is not None:
        with st.spinner(f"Processing and embedding {uploaded_file.name}..."):
            try:
                raw_text = get_document_text(uploaded_file)
                chunks = chunk_text(raw_text)
                build_vector_store(chunks, uploaded_file.name)
            except Exception as e:
                st.error(f"Error during document processing: {e}")
                
    st.markdown("---")
    st.markdown("Knowledge Base Status:")
    if st.session_state.knowledge_base["index"] is not None:
        num_chunks = len(st.session_state.knowledge_base["chunks"])
        st.success(f"✅ FAISS Index Active ({num_chunks} chunks)")
    else:
        st.warning("⚠️ No document loaded. Chat will be general conversation.")
        
    if st.button("Clear History & Documents"):
        # NEW: Added removal of the persistent history file
        for file_path in [INDEX_FILE, METADATA_FILE, HISTORY_FILE]:
            if os.path.exists(file_path):
                 os.remove(file_path)
             
        st.session_state.messages = []
        st.session_state.knowledge_base = {"index": None, "chunks": [], "filenames": []}
        st.experimental_rerun()


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user chat input
if prompt := st.chat_input("Ask a question..."):
    handle_user_input(prompt)