import streamlit as st
from langchain_ollama import OllamaLLM

# Initialize the LLM using the local model
MODEL_NAME = "llama3.1:latest"  # Ensure this matches your installed model
llm = OllamaLLM(model=MODEL_NAME)

# Streamlit chatbot interface
st.title("Local Model Chatbot")
st.write("This chatbot is powered by your local model. Ask me anything!")

# Store chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input from user
user_input = st.text_input("Enter your question:")
if user_input:
    # Save user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Generate response from LLM
    response = llm.invoke(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display the chat history
st.markdown("### Chat History")
for message in st.session_state.chat_history:
    role = "You" if message["role"] == "user" else "Bot"
    st.markdown(f"**{role}:** {message['content']}")
