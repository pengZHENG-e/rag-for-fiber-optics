from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_community.document_loaders import PDFMinerLoader


file_path = "data/fiber-optic-distributed-temperature-analysis-book.pdf"
loader = PyPDFLoader(file_path, extract_images=False)
# loader = PyMuPDFLoader(file_path)
# loader = PDFMinerLoader(file_path)


def load(path):
    loader = PyPDFLoader(path)
    return loader.load_and_split()


if __name__ == "__main__":
    # chunks = loader.load_and_split()
    chunks = [doc.page_content for doc in loader.load()]
    # pages = loader.load()    
    print(chunks[0])

