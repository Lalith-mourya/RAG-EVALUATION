import os
import re
import glob

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 
from langchain_chroma import Chroma

load_dotenv()

DATA_DIR = "DATA"
DB_DIR = "chroma_store"

# ACT as load_data() from document_loaders
def load_transcripts():
    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line=="WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)
        session = re.search(r"Session[ _]*(\d+)", path).group(1)
        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


def load_store():
    embeddings = HuggingFaceEmbeddings()

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    ).split_documents(docs)

    return Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_DIR
    )

def build_retriever():
    return load_store().as_retriever(
    search_kwargs={"k": 5}
)

if __name__ == "__main__":
    
    retriever = build_retriever()
    results = retriever.invoke("What is meant by regression testing?")
    for i in results:
        print(f"[Session {i.metadata['session']}] {i.page_content[:150]}...\n")
    