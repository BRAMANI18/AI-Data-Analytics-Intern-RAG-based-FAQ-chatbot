# ======================================================
# STEP 1: IMPORT BASICS
# ======================================================
import os
from dotenv import load_dotenv  # To load API keys from .env

# ======================================================
# STEP 2: IMPORT PINECONE
# ======================================================
from pinecone import Pinecone, ServerlessSpec  
# Pinecone = client to talk to Pinecone DB
# ServerlessSpec = only needed if creating new index

# ======================================================
# STEP 3: IMPORT LANGCHAIN
# ======================================================
from langchain_pinecone import PineconeVectorStore  
# This is the "glue" between text, embeddings, and Pinecone

from langchain_huggingface.embeddings import HuggingFaceEmbeddings  
# Converts text → vectors automatically using HF model

from langchain_core.documents import Document  
# Represents your text + optional metadata

# ======================================================
# STEP 4: LOAD API KEYS
# ======================================================
load_dotenv()  
# Loads PINECONE_API_KEY and HUGGING_FACE_API_KEY from .env

# ======================================================
# STEP 5: CONNECT TO PINECONE
# ======================================================
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))  
# Initializes Pinecone client

index_name = "sample-index"  
index = pc.Index(index_name)  
# References existing Pinecone index
# Dimension, metric (cosine), cloud, region are already set

# ======================================================
# STEP 6: INITIALIZE EMBEDDING MODEL
# ======================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)  
# Converts any text → fixed-length vector automatically
# Captures semantic meaning so similar texts → nearby vectors

# ======================================================
# STEP 7: CONNECT EMBEDDINGS WITH PINECONE
# ======================================================
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)
# WHAT THIS DOES:
# 1. Knows which Pinecone index to use for storage/search
# 2. Knows how to convert text → vectors using Hugging Face
# 3. Enables adding/searching text directly without manually handling vectors




# ======================================================
# STEP 9: RETRIEVE DOCUMENTS (QUERY)
# ======================================================
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.6}
)
# Converts query → vector automatically
# Searches Pinecone → finds nearest semantic matches
# Filters results based on similarity threshold

query = "what did you have for breakfast?"
results = retriever.invoke(query)

# ======================================================
# STEP 10: DISPLAY RESULTS
# ======================================================
print("RESULTS:\n")
for res in results:
    print(f"* {res.page_content}")
    print(f"  Metadata: {res.metadata}\n")

# ======================================================
# ANALOGY TO REMEMBER
# ======================================================
# Embedding model → Language translator (text → vector)
# Pinecone → Warehouse (stores vectors efficiently)
# PineconeVectorStore → Logistics manager
#    Converts text → vector, stores in warehouse, retrieves later
