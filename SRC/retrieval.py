import os
import re
import glob

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core import documents

load_dotenv()

DATA_DIR = "../DATA"
DB_DIR = "chroma_store"

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
        docs.append(text)
    return docs

if __name__ == "__main__":
    
    docs = load_transcripts()
    