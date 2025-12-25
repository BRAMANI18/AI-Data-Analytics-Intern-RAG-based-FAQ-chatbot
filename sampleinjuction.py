# =========================
# IMPORT BASICS
# =========================
import os
import time
from dotenv import load_dotenv

# =========================
# PINECONE
# =========================
from pinecone import Pinecone, ServerlessSpec

# =========================
# LANGCHAIN
# =========================
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# INIT PINECONE
# =========================
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "sample-index"

# =========================
# CREATE INDEX IF NOT EXISTS
# =========================
existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # MUST match embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

    # wait until ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# =========================
# INIT HUGGING FACE EMBEDDINGS
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# VECTOR STORE
# =========================
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# =========================
# DOCUMENTS
# =========================
documents = [
    Document(
        page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website"},
    ),
    Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website"},
    ),
    Document(
        page_content="LangGraph is the best framework for building stateful, agentic applications!",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="The stock market is down 500 points today due to fears of a recession.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet"},
    ),
]

# =========================
# ADD DOCUMENTS TO PINECONE
# =========================
vector_store.add_documents(documents)

print("✅ Documents successfully ingested into Pinecone using Hugging Face embeddings")
