# ======================================================
# IMPORT BASICS
# ======================================================
import os
import time
import uuid
from dotenv import load_dotenv

# ======================================================
# IMPORT PINECONE
# ======================================================
from pinecone import Pinecone, ServerlessSpec

# ======================================================
# IMPORT LANGCHAIN + HUGGING FACE EMBEDDINGS
# ======================================================
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings

# DOCUMENT LOADING & SPLITTING
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ======================================================
# LOAD ENV VARIABLES
# ======================================================
load_dotenv()

# ======================================================
# INITIALIZE PINECONE
# ======================================================
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")

existing_indexes = [info["name"] for info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,  # dimension must match embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # wait until the index is ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# ======================================================
# INITIALIZE HUGGING FACE EMBEDDINGS + VECTOR STORE
# ======================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# ======================================================
# LOAD PDF DOCUMENTS
# ======================================================
loader = PyPDFDirectoryLoader("documents/")  # make sure folder exists
raw_documents = loader.load()
print(f"PDF files loaded: {len(raw_documents)}")

# ======================================================
# SPLIT DOCUMENTS
# ======================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len
)

documents = text_splitter.split_documents(raw_documents)
print(f"Text chunks created: {len(documents)}")

# ======================================================
# GENERATE UNIQUE IDS
# ======================================================
uuids = [str(uuid.uuid4()) for _ in documents]

# ======================================================
# STORE IN PINECONE
# ======================================================
vector_store.add_documents(documents=documents, ids=uuids)

print("✅ Documents successfully stored in Pinecone")
