import os
from langchain_chroma import Chroma
from langchain_community import embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def initialize_embeddings():
    embd = embeddings.OllamaEmbeddings(model="nomic-embed-text:latest", num_gpu=8)
    return embd

def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return ' '.join([doc.page_content for doc in documents])

def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents([text])

def create_vector_store(chunks):
    embeddings = initialize_embeddings()
    return Chroma.from_documents(chunks, embeddings)

def start_vector_store(file_path):
    if not os.path.exists(file_path):
        return None
    raw_text = extract_text_from_pdf(file_path)
    chunks = split_text_into_chunks(raw_text)
    vector_store = create_vector_store(chunks)
    return vector_store