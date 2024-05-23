import sys
sys.path.append(".")

import rag.LLMs as LLM
import rag.chat_memory as memory
import rag.prompts as prompts
from rag.vectror_DB.faiss_search import create_embeddings, load_faiss_index, search_index, get_pages, get_sources, get_metasource

from sentence_transformers import CrossEncoder


def HyDE_chat(query, faiss_db, tokenizer, model):
    messages = [{"role": "user", "content": query}]
    formatted_input = prompts.get_formatted_input_HyDE(messages)
    hyde_answer = LLM.Llama3_ChatQA(tokenizer, model, formatted_input)
    hyde_query = query + hyde_answer
    try:
        hyde_docs  = search_index(faiss_db, hyde_query, k = 20)
        # context = get_pages(hyde_docs)
        # context = str(context)
        # pages = get_metasource(hyde_docs)
        # print(pages)
    except Exception as e:
        context = "No document extracted."
    cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, device="cuda"
    )
    reranked_docs = cross_encoder.rank(
        query,
        [doc.page_content for doc in hyde_docs],
        top_k=5,
        return_documents=True,
    )
    context = "".join(doc["text"] + "\n" for doc in reranked_docs)

    messages = [{"role": "user", "content": query}]

    formatted_input = prompts.get_formatted_input(messages, context)

    response = LLM.Llama3_ChatQA(tokenizer, model, formatted_input)    

    return response


def chat(query, faiss_db, tokenizer, model):
    try:
        retrieved_docs  = search_index(faiss_db, query, k = 5)
        context = get_pages(retrieved_docs)
        context = str(context)
        pages = get_metasource(retrieved_docs)
        print(pages)
    except Exception as e:
        context = "No document extracted."
    # cross_encoder = CrossEncoder(
    #     "cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, device="cuda"
    # )
    # reranked_docs = cross_encoder.rank(
    #     query,
    #     [doc.page_content for doc in retrieved_docs],
    #     top_k=3,
    #     return_documents=True,
    # )
    # context = "".join(doc["text"] + "\n" for doc in reranked_docs)

    user_message = {"role": "user", "content": query}
    messages = [user_message]

    formatted_input = prompts.get_formatted_input(messages, context)

    response = LLM.Llama3_ChatQA(tokenizer, model, formatted_input)    

    return response


def chat_with_memory(query, faiss_db, tokenizer, model, memory):
    try:
        retrieved_docs  = search_index(faiss_db, query, k =3)
        # context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        context = get_pages(retrieved_docs)
        context = str(context)
        print(context)
    except Exception as e:
        context = "No document extracted."
    
    messages = [{"role": "user", "content": query}]
    
    formatted_input = prompts.get_formatted_input_with_memory(messages, context, memory)

    response = LLM.Llama3_ChatQA(tokenizer, model, formatted_input)    

    return response


def chat_test(folder_path="data/faiss_db/chunk-256", index_name="FO_bge-base-en-v1.5-chunk(256)"):
    model_path = "BAAI/bge-base-en-v1.5"
    embeddings = create_embeddings(model_path, device='cuda', normalize_embeddings=False)
    faiss_db = load_faiss_index(folder_path=folder_path, index_name=index_name, embeddings=embeddings, allow_dangerous_deserialization=True)
    tokenizer, model = LLM.init_Llama3_ChatQA()
    

    query1 = "What is distributed temperature sensing (DTS) and how does it measure temperature along an optical fiber?"
    query2 = "What is the length of the fiber installed in the 14,000-ft well?"

    while True:
        user_input = input("Enter what you want to say: ")
        response = HyDE_chat(user_input, faiss_db, tokenizer, model)
        print(response)

def chat_with_memory_test(folder_path="data/faiss_db", index_name="FO"):
    model_path = "BAAI/bge-base-en-v1.5"
    embeddings = create_embeddings(model_path, device='cuda', normalize_embeddings=False)
    faiss_db = load_faiss_index(folder_path=folder_path, index_name=index_name, embeddings=embeddings, allow_dangerous_deserialization=True)
    tokenizer, model = LLM.init_Llama3_ChatQA()

    query1 = "What is distributed temperature sensing (DTS) and how does it measure temperature along an optical fiber?"
    query2 = "What is the length of the fiber installed in the 14,000-ft well?"

    hist = memory.Memory(k=5)

    while True:
        user_input = input("Enter what you want to say: ")
        response = chat_with_memory(user_input, faiss_db, tokenizer, model, hist.load())
        print(response)
        hist.save(user_input, response)
    
if __name__ == "__main__":
    chat_test()
