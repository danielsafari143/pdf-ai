import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import conversational_retrieval
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import openai


def process_pdf(file_path):
    """Process PDF with fallback strategies"""
    try:
        # Try different loaders with fallback
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        except:
            loader = UnstructuredPDFLoader(file_path, strategy="ocr_only")
            documents = loader.load()

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma.from_documents(documents, embeddings)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None
