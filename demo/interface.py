import sys, os
sys.dont_write_bytecode = True

import time
from dotenv import load_dotenv

import pandas as pd
import streamlit as st
import openai
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_agent import ChatBot
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
FAISS_PATH = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

welcome_message = """
  #### Welcome to Smart Resume Matcher! 🎯

  This AI-powered system helps hiring managers find the best candidates from thousands of resumes quickly and accurately.

  **How it works:**
  - Upload your resume database
  - Describe the job requirements in natural language  
  - Get intelligent candidate recommendations with detailed analysis

  **Getting started:**
  1. Add your OpenAI API key in the sidebar 🔑
  2. Upload your resume CSV file (with 'ID' and 'Resume' columns) 📁
  3. Start asking about candidates or job requirements 💬

  **Example queries:**
  - "Find software engineers with Python experience"
  - "Show me details for applicant ID 123" 
  - "Compare the top 3 candidates for this role"
"""

st.set_page_config(
    page_title="Smart Resume Matcher", 
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Smart Resume Matcher")
# st.markdown("*AI-Powered Intelligent Candidate Screening*")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content=welcome_message)]

if "df" not in st.session_state:
    try:
        if os.path.exists(DATA_PATH):
            st.session_state.df = pd.read_csv(DATA_PATH)
        else:
            st.session_state.df = None
    except Exception as e:
        st.session_state.df = None
        st.error(f"Error loading default data: {str(e)}")

if "embedding_model" not in st.session_state:
    try:
        st.session_state.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, 
            model_kwargs={"device": "cpu"}
        )
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")

if "rag_pipeline" not in st.session_state and st.session_state.df is not None:
    try:
        if os.path.exists(FAISS_PATH):
            vectordb = FAISS.load_local(
                FAISS_PATH, 
                st.session_state.embedding_model, 
                distance_strategy=DistanceStrategy.COSINE, 
                allow_dangerous_deserialization=True
            )
            st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
        else:
            st.session_state.rag_pipeline = None
    except Exception as e:
        st.session_state.rag_pipeline = None
        st.warning("Vector database not found. Please upload resume data to create one.")

if "resume_list" not in st.session_state:
    st.session_state.resume_list = []

def upload_file():
    if st.session_state.uploaded_file is not None:
        try:
            with st.spinner("Processing your resume data..."):
                df_load = pd.read_csv(st.session_state.uploaded_file)
                
                # Validate required columns
                if "Resume" not in df_load.columns or "ID" not in df_load.columns:
                    st.error("❌ Missing required columns. Your CSV must have 'Resume' and 'ID' columns.")
                    return
                
                if df_load.empty:
                    st.error("❌ The uploaded file is empty")
                    return
                
                # Create vector store
                with st.spinner("Creating search index... This may take a few minutes"):
                    vectordb = ingest(df_load, "Resume", st.session_state.embedding_model)
                    st.session_state.df = df_load
                    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, df_load)
                
                st.success(f"✅ Successfully loaded {len(df_load)} resumes!")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Please ensure your file is a valid CSV with 'Resume' and 'ID' columns")
    else:
        # Reset to default data
        try:
            if os.path.exists(DATA_PATH):
                st.session_state.df = pd.read_csv(DATA_PATH)
                if os.path.exists(FAISS_PATH):
                    vectordb = FAISS.load_local(
                        FAISS_PATH, 
                        st.session_state.embedding_model, 
                        distance_strategy=DistanceStrategy.COSINE, 
                        allow_dangerous_deserialization=True
                    )
                    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)
        except Exception as e:
            st.error(f"Error loading default data: {str(e)}")

def check_openai_api_key(api_key: str):
    openai.api_key = api_key
    try:
        _ = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=3
        )
        return True
    except openai.AuthenticationError:
        return False
    except Exception:
        return True

def check_model_name(model_name: str, api_key: str):
    openai.api_key = api_key
    try:
        model_list = [model.id for model in openai.models.list()]
        return model_name in model_list
    except:
        return True

def clear_message():
    st.session_state.resume_list = []
    st.session_state.chat_history = [AIMessage(content=welcome_message)]

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Show system status
    if st.session_state.df is not None:
        st.success(f"📊 {len(st.session_state.df)} resumes loaded")
    else:
        st.warning("📊 No resume data loaded")
    
    st.divider()
    
    # API Key
    api_key = st.text_input("OpenAI API Key", type="password", key="api_key")
    if api_key:
        if check_openai_api_key(api_key):
            st.success("✅ API Key Valid")
        else:
            st.error("❌ Invalid API Key")
    
    # Model settings
    st.selectbox("AI Mode", ["Generic RAG", "RAG Fusion"], key="rag_selection")
    st.selectbox("AI Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], key="gpt_selection")
    
    st.divider()
    
    # File upload
    st.subheader("📁 Upload Resumes")
    st.file_uploader(
        "Upload CSV file", 
        type=["csv"], 
        key="uploaded_file", 
        on_change=upload_file,
        help="CSV file with 'ID' and 'Resume' columns"
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        st.button("🗑️ Clear Chat", on_click=clear_message)
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    st.divider()
    
    # Help section
    with st.expander("💡 Help & Tips"):
        st.markdown("""
        **Quick Start:**
        1. Add OpenAI API key above
        2. Upload resume CSV or use sample data
        3. Ask about candidates or job requirements
        
        **Example Questions:**
        - "Find Python developers with 3+ years experience"
        - "Show me applicant ID 42"
        - "Who are the best candidates for a marketing role?"
        """)

# Main chat interface
user_query = st.chat_input("Ask about candidates, job requirements, or specific applicants...")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            message[0].render(*message[1:])

# Process user input
if user_query and user_query.strip():
    # Validation
    if not st.session_state.get('api_key'):
        st.warning("🔑 Please add your OpenAI API key in the sidebar to continue.")
        st.stop()
    
    if not check_openai_api_key(st.session_state.api_key):
        st.error("❌ Invalid OpenAI API key. Please check your key and try again.")
        st.stop()
    
    if not st.session_state.rag_pipeline:
        st.warning("⚠️ Please upload resume data first.")
        st.stop()
    
    # Add user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            start = time.time()
            
            with st.spinner("Analyzing your request..."):
                retriever = st.session_state.rag_pipeline
                llm = ChatBot(
                    api_key=st.session_state.api_key,
                    model=st.session_state.gpt_selection,
                )
                
                document_list = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection)
                query_type = retriever.meta_data["query_type"]
                st.session_state.resume_list = document_list
                stream_message = llm.generate_message_stream(user_query, document_list, [], query_type)
            
            response = st.write_stream(stream_message)
            end = time.time()
            
            # Add to chat history
            st.session_state.chat_history.append(AIMessage(content=response))
            
            # Show processing details
            retriever_message = chatbot_verbosity
            retriever_message.render(document_list, retriever.meta_data, end-start)
            st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end-start))
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Please try rephrasing your question or check your settings.")