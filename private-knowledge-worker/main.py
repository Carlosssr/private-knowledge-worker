"""
main.py  —  Run with:  python main.py
Private RAG assistant over Docs/ and Spreadsheets/ folders.
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import os, glob
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# LLM providers
from langchain_ollama import ChatOllama          # pip install -U langchain-ollama
from langchain_openai import ChatOpenAI          # pip install -U langchain-openai

import gradio as gr

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
USE_OPENAI      = False           # True → GPT‑4o‑mini (needs OPENAI_API_KEY)
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))   # /Main
ROOT_KB         = os.path.join(BASE_DIR, "..")

DOCS_FOLDER     = os.path.join(ROOT_KB, "Docs")
SHEET_FOLDER    = os.path.join(ROOT_KB, "Spreadsheets")
VECTOR_DIR      = os.path.join(ROOT_KB, "vector_db")

CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
AUTH_CREDENTIAL = ("me", "mypassword")
LLM_MODEL       = "gpt-4o-mini" if USE_OPENAI else "llama3"

load_dotenv(override=True)        # pulls OPENAI_API_KEY if needed

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def add_type(doc: Document, doc_type: str) -> Document:
    doc.metadata["doc_type"] = doc_type
    return doc

def load_csv_rows(path: str, doc_type: str) -> list[Document]:
    """Create one Document per row so the model can reference values precisely."""
    df = pd.read_csv(path, low_memory=False, dtype=str)
    docs = []
    for i, row in df.iterrows():
        text = " | ".join(f"{col}: {val}" for col, val in row.items())
        docs.append(Document(page_content=text,
                             metadata={"source": path,
                                       "row_index": i,
                                       "doc_type": doc_type}))
    return docs

def load_xlsx_rows(path: str, doc_type: str) -> list[Document]:
    docs = []
    sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    for sheet_name, df in sheets.items():
        for i, row in df.iterrows():
            text = " | ".join(f"{col}: {val}" for col, val in row.items())
            docs.append(Document(page_content=text,
                                 metadata={"source": path,
                                           "sheet": sheet_name,
                                           "row_index": i,
                                           "doc_type": doc_type}))
    return docs

# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
documents: list[Document] = []

# Docs folder
if os.path.isdir(DOCS_FOLDER):
    loader = DirectoryLoader(
        DOCS_FOLDER,
        glob="**/*.*",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents += [add_type(d, "docs") for d in loader.load()]

# CSV rows
for csv_path in glob.glob(f"{SHEET_FOLDER}/**/*.csv", recursive=True):
    documents += load_csv_rows(csv_path, "sheet_row")

# XLSX rows
for xlsx_path in glob.glob(f"{SHEET_FOLDER}/**/*.xlsx", recursive=True):
    documents += load_xlsx_rows(xlsx_path, "sheet_row")

print(f"Loaded {len(documents)} raw docs (rows & text files).")

# --------------------------------------------------------------------------- #
# Split & embed
# --------------------------------------------------------------------------- #
splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                 chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
print(f"Chunks: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Reset vector store if exists
if os.path.exists(VECTOR_DIR):
    Chroma(persist_directory=VECTOR_DIR,
           embedding_function=embeddings).delete_collection()

vector_db = Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_DIR)
print("DB ready with", vector_db._collection.count(), "vectors")

# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
if USE_OPENAI:
    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0.2)
else:
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2,
                     base_url="http://localhost:11434")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(k=10),
    memory=memory
)

# --------------------------------------------------------------------------- #
# Gradio
# --------------------------------------------------------------------------- #
def chat_fn(message: str, history):
    """Answer a question using RAG."""
    return qa_chain.invoke({"question": message})["answer"]

iface = gr.ChatInterface(chat_fn,
                         title="🔒 Private Knowledge Worker",
                         type="messages")

iface.launch(inbrowser=True, auth=AUTH_CREDENTIAL)
