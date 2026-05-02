import os
import streamlit as st
from dotenv import load_dotenv
 
# LangChain + Hugging Face + Groq
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
 
# Load environment variables
load_dotenv()
 
# Hugging Face token
HUGGINGFACEHUB_API_TOKEN = st.secrets["HF_TOKEN"]
st.secrets["HF_TOKEN"]
 
# Streamlit UI
st.title("Conversational RAG with PDF Uploads")
st.write("Upload PDFs and chat with their content using Hugging Face embeddings + Groq LLM")
 
# Groq API key input
api_key = st.text_input("Enter your Groq API key:", type="password")
 
if api_key:
    # Initialize Groq LLM (use supported model)
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
 
    # Session management
    session_id = st.text_input("Session ID", value="default_session")
    if "store" not in st.session_state:
        st.session_state.store = {}
 
    # File uploader
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
 
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
 
        # Split and embed
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
 
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
 
        # Prompt template
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Use retrieved context to answer concisely:\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
 
        def rag_chain(question, chat_history):
            retrieved_docs = retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            prompt = qa_prompt.format(chat_history=chat_history, input=question, context=context)
            return llm.invoke(prompt)
 
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
 
        # User input
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = rag_chain(user_input, session_history.messages)
            session_history.add_user_message(user_input)
            session_history.add_ai_message(response.content)
 
            st.write("Assistant:", response.content)
            st.write("Chat History:", session_history.messages)
 
else:
    st.warning("Please enter your Groq API Key")
