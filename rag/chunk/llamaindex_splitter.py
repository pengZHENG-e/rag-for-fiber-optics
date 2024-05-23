from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

def split(paths, chunk_size=200, chunk_overlap=100):
    documents = SimpleDirectoryReader(input_files = paths).load_data()
    # print(len(documents))
    # print(f"Document Metadata: {documents[0].metadata}")

    splitter = SentenceSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    # print(f"Length of nodes : {len(nodes)}")
    # print(f"get the content for node 0 :{nodes[100].get_content(metadata_mode='all')}")
    return nodes

if __name__ == "__main__":
    paths = ["data/fiber-optic-distributed-temperature-analysis-book.pdf"]
    nodes = split(paths, chunk_size=200, chunk_overlap=100)
    print(nodes[0].get_content())