import sys
sys.path.append(".")
from rag.utils import get_paths

from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


def load_dataset(dataset_name, page_content_column):
    """
    Load dataset from Hugging Face and return the data.
    """
    loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
    data = loader.load()
    return data

def split_docs(data, chunk_size, chunk_overlap):
    """
    Split text data into chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(data)
    return docs

def create_embeddings(model_path, device='cuda', normalize_embeddings=False):
    """
    Create embeddings using HuggingFaceEmbeddings.
    """
    model_kwargs = {'device': device, 'trust_remote_code':True}
    encode_kwargs = {'normalize_embeddings': normalize_embeddings}
    embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    return embeddings

def create_faiss_index(docs, embeddings, folder_path, index_name):
    """
    Create FAISS index and save it locally.
    """
    try:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(folder_path=folder_path, index_name=index_name)
        print("Faiss index created")
    except Exception as e:
        print(f"Faiss store failed \n{e}")


def add_faiss_index(docs, embeddings, folder_path, index_name, allow_dangerous_deserialization=True):    
    db = FAISS.load_local(folder_path=folder_path, embeddings=embeddings, index_name=index_name, allow_dangerous_deserialization=allow_dangerous_deserialization)
    db.add_documents(documents = docs)
    db.save_local(folder_path=folder_path, index_name=index_name)
    print("Faiss index added")


def create_or_add(pdf_paths, model_path, folder_path, index_name, chunk_size = 512, chunk_overlap= 100):
    _docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path, extract_images=False)
        docs = loader.load()
        docs = split_docs(docs, chunk_size = chunk_size, chunk_overlap=chunk_overlap)
        for doc in docs:
            _docs.append(doc)

    embeddings = create_embeddings(model_path, device='cuda', normalize_embeddings=False)
    try:
        add_faiss_index(_docs, embeddings, folder_path, index_name, allow_dangerous_deserialization=True)
    except Exception as e:
        create_faiss_index(_docs, embeddings, folder_path, index_name)

def main():
    file_paths = get_paths("data", ".pdf")
    model_path = "BAAI/bge-large-en-v1.5"
    folder_path = "data/faiss_db/chunk-256"
    index_name = "FO_bge-large-en-v1.5-chunk(256)"
    create_or_add(file_paths, model_path, folder_path, index_name, chunk_size = 256, chunk_overlap= 100)

if __name__ == "__main__":
    main()
