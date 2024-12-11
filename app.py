import os
import numpy as np
import streamlit as st
import faiss
import PyPDF2
import docx
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

# Paths and Configuration
UPLOAD_DIRECTORY = "/home/mazo/Mazo_Solutions/Code_Local/LLM_Integration/uploaded_docs/"
FAISS_INDEX_PATH = "/home/mazo/Mazo_Solutions/Code_Local/LLM_Integration/faiss_index"
EMBEDDING_DIM = 384  # Dimension for "all-MiniLM-L6-v2"
MODEL_NAME = "llama3.1:latest"

# Ensure upload directory exists
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Initialize the LLM
llm = OllamaLLM(model=MODEL_NAME)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize or load FAISS index
def initialize_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    else:
        return faiss.IndexFlatL2(EMBEDDING_DIM)

faiss_index = initialize_faiss_index()
docstore = {}  # Maps document IDs to content
index_to_doc_id = {}  # Maps FAISS indices to document IDs

# Function to extract text from uploaded files
def extract_text(file_path):
    ext = file_path.split('.')[-1].lower()
    try:
        if ext == "pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([page.extract_text() for page in reader.pages])
        elif ext == "docx":
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            st.error("Unsupported file type!")
            return None
    except Exception as e:
        st.error(f"Error reading {file_path}: {e}")
        return None

# Index new documents
def index_documents():
    documents = [f for f in os.listdir(UPLOAD_DIRECTORY) if os.path.isfile(os.path.join(UPLOAD_DIRECTORY, f))]
    for doc in documents:
        file_path = os.path.join(UPLOAD_DIRECTORY, doc)
        content = extract_text(file_path)
        if content and file_path not in docstore:
            embedding = np.array(embeddings.embed_documents([content])).astype(np.float32)
            if embedding.shape[1] == EMBEDDING_DIM:
                faiss_index.add(embedding)
                docstore[file_path] = content
                index_to_doc_id[len(index_to_doc_id)] = file_path
                faiss.write_index(faiss_index, FAISS_INDEX_PATH)
                st.sidebar.success(f"Document '{doc}' indexed successfully!")

# Initialize Streamlit app
st.title("Chatbot with RAG")
st.sidebar.title("Upload Documents for RAG")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or Word files", type=["pdf", "docx"], accept_multiple_files=True
)

# Handle file uploads
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    index_documents()

# Process already uploaded documents
if not uploaded_files:
    index_documents()

# Chat functionality
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask me a question:")
if user_input:
    # Save query to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Embed the user query
    query_embedding = np.array(embeddings.embed_documents([user_input])).astype(np.float32)
    distances, indices = faiss_index.search(query_embedding, k=1)

    # Retrieve relevant document content
    relevant_content = ""
    reference = ""
    if indices[0][0] != -1 and distances[0][0] < 1.0:  # Threshold for relevance
        doc_id = index_to_doc_id.get(indices[0][0], "")
        relevant_content = docstore.get(doc_id, "")
        reference = f"**Reference:** Content derived from '{doc_id}'"

    # Generate response using the LLM
    if relevant_content:
        dynamic_prompt = f"Given the following information:\n\n{relevant_content}\n\nAnswer the question: {user_input}"
        response = llm.invoke(dynamic_prompt)
    else:
        # Out-of-context query, directly use LLM without document context
        response = llm.invoke(user_input)
        reference = "**Note:** No relevant document found; the answer is based on general knowledge."

    # Append LLM's response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response + "\n\n" + reference})


# Display chat history
st.markdown("### Chat History")
for message in st.session_state.chat_history:
    role = "You" if message["role"] == "user" else "Bot"
    st.markdown(f"**{role}:** {message['content']}")
