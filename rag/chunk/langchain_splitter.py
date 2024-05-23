
from tqdm.notebook import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# RAW_KNOWLEDGE_BASE = [
#     LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
# ]
# print(RAW_KNOWLEDGE_BASE[99].page_content)


MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # the maximum number of characters in a chunk: we selected this value arbitrarily
    chunk_overlap=100,  # the number of characters to overlap between chunks
    add_start_index=True,  # If `True`, includes chunk's start index in metadata
    strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
    separators=MARKDOWN_SEPARATORS,
)

EMBEDDING_MODEL_NAME = "thenlper/gte-small"


def load_str(text):
    doc =  LangchainDocument(page_content=text, metadata={"source": "local"})
    return [doc]


def split_documents(
    chunk_size: int,
    knowledge_base: list[LangchainDocument],
    tokenizer_name = EMBEDDING_MODEL_NAME,
) -> list[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        # chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

if __name__ == "__main__":
    #Test text_splitter
    # loader = TextLoader("data/text_example_2.txt")
    # RAW_KNOWLEDGE_BASE = loader.load()
    # print(RAW_KNOWLEDGE_BASE[0].page_content)
    # print(len(text))
    # docs_processed = []
    # for doc in RAW_KNOWLEDGE_BASE:
    #     docs_processed += text_splitter.split_documents([doc])

    # print("*"*100)
    # for doc in docs_processed:
    #     print(doc.page_content)
    #     print("*"*100)
    # print(len(docs_processed))


    #TEST split_documents()
    rag_dataset = load_dataset("neural-bridge/rag-dataset-1200")
    text = load_str(str(rag_dataset["train"]["context"][2]))
    docs_processed = split_documents(
        512,  
        text,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )

    print("*"*100)
    for doc in docs_processed:
        print(doc.page_content)
        print("*"*100)
    print(len(docs_processed))
