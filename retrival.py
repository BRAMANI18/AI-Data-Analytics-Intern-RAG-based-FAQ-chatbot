# ================================
# IMPORT BASICS
# ================================
import os
from dotenv import load_dotenv

# ================================
# IMPORT PINECONE
# ================================
from pinecone import Pinecone

# ================================
# IMPORT LANGCHAIN
# ================================
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ================================
# INIT PINECONE
# ================================
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

# ================================
# EMBEDDINGS (HUGGING FACE)
# ================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# ================================
# RETRIEVER (HF-COMPATIBLE)
# ================================
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5, "score_threshold": 0.1}
)

# ================================
# QUERY
# ================================
query = "What is Retrieval-Augmented Generation?"
results = retriever.invoke(query)

# ================================
# SHOW RESULTS
# ================================
print("\nRESULTS:\n")

if not results:
    print("No results found.")
else:
    for i, res in enumerate(results, 1):
        print(f"[{i}]")
        print(res.page_content[:700])
        print("METADATA:", res.metadata)
        print("-" * 80)
