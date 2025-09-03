## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
import os
import chromadb

st.set_page_config(page_title="Conversational RAG With Chat History", layout="wide")

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# --- DATABASE CONFIGURATION ---
# Define a path for the persistent Chroma database
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "pdf_store"

# Initialize embeddings once using Streamlit's caching
@st.cache_resource
def get_embeddings():
    """Returns a cached instance of the HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_chroma_client():
    """Returns a cached instance of the ChromaDB persistent client."""
    return chromadb.PersistentClient(path=CHROMA_PATH)

embeddings = get_embeddings()
client = get_chroma_client()

## Main title for the app
st.title("Conversational RAG With PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content. Your vector store is saved locally and will persist.")

## Input the Groq API Key in the sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Groq API key:", type="password")
    if not api_key:
        st.warning("Please enter your Groq API Key to proceed.")
        st.stop()

    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    ## File uploader in the sidebar
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

## Statefully manage chat history and the RAG chain
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'conversational_rag_chain' not in st.session_state:
    st.session_state.conversational_rag_chain = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# Process uploaded PDFs only if they are new
if uploaded_files:
    file_identifiers = [(file.name, file.size) for file in uploaded_files]
    if st.session_state.get('processed_files') != file_identifiers:
        st.session_state.processed_files = file_identifiers
        with st.spinner("Processing uploaded PDFs... This may take a moment."):
            documents = []
            temp_dir = "./temp_files"
            os.makedirs(temp_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                temppdf_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temppdf_path, "wb") as file:
                    file.write(uploaded_file.getvalue())
                loader = PyPDFLoader(temppdf_path)
                docs = loader.load()
                documents.extend(docs)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            try:
                client.delete_collection(name=COLLECTION_NAME)
                st.sidebar.info(f"Cleared old collection: '{COLLECTION_NAME}'")
            except ValueError:
                 # This error occurs if the collection doesn't exist, which is fine.
                pass

            vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings,
                client=client, # Pass the persistent client
                collection_name=COLLECTION_NAME
            )

            retriever = vectorstore.as_retriever(search_kwargs={'k': 5})    

            # --- RAG Chain Definition ---
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
                )
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, just say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise.\n\n{context}"
                )
            qa_prompt = ChatPromptTemplate.from_messages(
                    [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
                )
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]
            
            st.session_state.conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            st.sidebar.success("PDFs processed successfully! You can now ask questions.")

# --- Chat Interface ---
st.subheader("Chat with your documents")

session_id = "default_chat_session"

# Display chat history from session state
if session_id in st.session_state.store:
    for msg in st.session_state.store[session_id].messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

# Get user input
if user_input := st.chat_input("Ask a question about your documents..."):
    if st.session_state.conversational_rag_chain is None:
        st.warning("Please upload PDF documents first.")
    else:
        with st.chat_message("human"):
            st.markdown(user_input)
        
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )

                st.markdown(response['answer'])
