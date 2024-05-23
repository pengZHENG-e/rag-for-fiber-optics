from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

def create_embeddings(model_path, device='cuda', normalize_embeddings=False):
    """
    Create embeddings using HuggingFaceEmbeddings.
    """
    model_kwargs = {'device': device, 'trust_remote_code':True}
    encode_kwargs = {'normalize_embeddings': normalize_embeddings}
    embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    return embeddings

def load_faiss_index(folder_path, index_name, embeddings, allow_dangerous_deserialization=False):
    """
    Load FAISS index from the local path.
    """
    try:
        db = FAISS.load_local(folder_path=folder_path, embeddings=embeddings, index_name=index_name, allow_dangerous_deserialization=allow_dangerous_deserialization)
        print("Faiss index loaded")
        return db
    except Exception as e:
        print(f"Faiss index loading failed \n{e}")
        return None

def search_index(db, query, k = 10):
    """
    Search the FAISS index for the given query and print the top result.
    """
    if db is not None:
        search_docs = db.similarity_search(query=query, k = k )
        if search_docs:
            return search_docs

        else:
            return "No relevant results found."
    else:
        raise Exception("FAISS index not loaded.")

def get_pages(docs: list):
    pages = []
    loader_map = {}

    # Load PDF content for each unique source
    for doc in docs:
        source = doc.metadata["source"]
        if source not in loader_map:
            loader_map[source] = PyPDFLoader(source, extract_images=False)

    # Extract page content for each document
    for doc in docs:
        source = doc.metadata["source"]
        page_num = doc.metadata["page"]
        loader = loader_map[source]
        page_content = loader.load()[page_num].page_content
        if page_content not in pages:
            pages.append(page_content)

    return pages

def get_sources(docs:list):
    sources = []
    for doc in docs:
        source = doc.metadata["source"]
        if source not in sources:
            sources.append(source)
    return sources

def get_metasource(docs:list):
    sources = []
    for doc in docs:
        meta = doc.metadata["source"]
        if meta not in sources:
            sources.append(meta)
    return sources


def main():
    model_path = "BAAI/bge-base-en-v1.5"
    embeddings = create_embeddings(model_path, device='cuda', normalize_embeddings=False)
    faiss_db = load_faiss_index(folder_path="data/faiss_db", index_name="FO_bge-base-en-v1.5", embeddings=embeddings, allow_dangerous_deserialization=True)

    while True:
        query = input("Enter your query: ")
        docs = search_index(faiss_db, query, k=30)
        print(docs)
        sources = get_sources(docs)
        print(sources)

if __name__ == "__main__":
    main()