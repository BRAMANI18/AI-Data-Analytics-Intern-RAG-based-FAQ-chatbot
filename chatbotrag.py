import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from transformers import pipeline
import uuid
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

st.title("Hugging Face RAG Chatbot")

# -----------------------------
# Initialize Pinecone index
# -----------------------------
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# -----------------------------
# Initialize Hugging Face embeddings + vector store
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks."))

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# -----------------------------
# User input
# -----------------------------
prompt = st.chat_input("Ask a question")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    # -----------------------------
    # Retriever
    # -----------------------------
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3},
    )
    docs = retriever.invoke(prompt)
    docs_text = "\n".join(d.page_content for d in docs)

    # -----------------------------
    # Hugging Face Generative QA
    # -----------------------------
    qa_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=200
    )

    input_text = f"Answer the question based on the context below. If the answer is not in the context, say 'I don't know'.\nContext: {docs_text}\nQuestion: {prompt}"
    answer = qa_pipeline(input_text)[0]['generated_text']

    # -----------------------------
    # Display answer
    # -----------------------------
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.session_state.messages.append(AIMessage(answer))
