import os
import numpy as np
import streamlit as st
import faiss
import PyPDF2
import docx
import csv
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Configuration
UPLOAD_DIRECTORY = "uploaded_docs"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_DIM = 384  # For "all-MiniLM-L6-v2" model
MODEL_NAME = "llama3.1:latest"  # Replace with your local model

# Ensure upload directory exists
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Initialize LLM
llm = OllamaLLM(
    model=MODEL_NAME,
    system_prompt=(
        "You are an expert IT support assistant. Always respond with a structured, step-by-step solution "
        "to resolve the user's problem. Focus on actionable steps, provide clarity, and ensure proper formatting "
        "with clean spacing and appropriate line breaks."
    )
)

# Initialize HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize FAISS Index
def initialize_vector_store():
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    else:
        return faiss.IndexFlatL2(EMBEDDING_DIM)

faiss_index = initialize_vector_store()
docstore = {}
index_to_docstore_id = {}

# Text Extraction Functions
def extract_text(file_path):
    file_ext = file_path.split('.')[-1].lower()
    try:
        if file_ext == "pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([page.extract_text() for page in reader.pages])
        elif file_ext == "docx":
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_ext == "txt":
            with open(file_path, "r") as f:
                return f.read()
        elif file_ext == "csv":
            with open(file_path, "r") as f:
                reader = csv.reader(f)
                return "\n".join([",".join(row) for row in reader])
        elif file_ext == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
    except Exception as e:
        st.error(f"Error extracting text from {file_path}: {e}")
        return ""

# Process and Index Documents
def index_documents():
    document_files = [f for f in os.listdir(UPLOAD_DIRECTORY) if os.path.isfile(os.path.join(UPLOAD_DIRECTORY, f))]
    for file_name in document_files:
        file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
        content = extract_text(file_path)
        if content:
            embedding = np.array(embeddings.embed_documents([content])).astype(np.float32)
            if embedding.shape[1] == EMBEDDING_DIM:
                faiss_index.add(embedding)
                docstore[file_path] = content
                index_to_docstore_id[len(index_to_docstore_id)] = file_path
                faiss.write_index(faiss_index, FAISS_INDEX_PATH)
                st.sidebar.info(f"Document '{file_name}' indexed successfully!")

# Sidebar for File Uploads
st.sidebar.title("Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload documents", type=["pdf", "docx", "txt", "csv", "json"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    index_documents()

# Process Existing Documents if No Uploads
if not uploaded_files:
    index_documents()

# Chat Interface
st.title("Mazo's TroubleFix")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask me anything")
if user_input:
    # Store user query
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Query FAISS for relevant content
    query_embedding = np.array(embeddings.embed_documents([user_input])).astype(np.float32)
    distances, indices = faiss_index.search(query_embedding, k=1)

    relevant_content = ""
    reference = ""
    if indices[0][0] != -1:
        relevant_doc_id = index_to_docstore_id.get(indices[0][0], "")
        relevant_content = docstore.get(relevant_doc_id, "")
        if relevant_content:
            response = f"Answer from document: {relevant_content}"
            reference = f"\n\n**Reference:** Derived from document '{relevant_doc_id}'"

    # If no relevant content is found, use the model to generate a response
    if not relevant_content:
        dynamic_prompt = f"How to solve '{user_input}'?"
        response = llm.invoke(dynamic_prompt)

    # Append response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response + reference})

# Display Chat History
st.markdown("### Chat History")
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}")
    else:
        st.markdown(f"**Response:** {chat['content']}")