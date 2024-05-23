import sys
sys.path.append(".")

from rag.scripts.embedders import search
from rag.chunk.langchain_splitter import load_str, split_documents
from rag.LLMs import llama3_8b_ins

import time
import random

from datasets import load_dataset

import torch

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForCausalLM


def write(path, text):
    try:
        with open(path, 'a') as file:
            file.write(text)
    except FileNotFoundError:
        with open(path, 'w') as file:
            file.write(text)


def chunk(rag_dataset):
    text = load_str(str(rag_dataset["train"]["context"]))
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    docs_processed = split_documents(
        512,  
        text,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )
    chunks = [doc.page_content for doc in docs_processed]
    return chunks

def filter(query, matches, llm_tokenizer, llm_model):
    FILTER_PROMPT = """\
    Query:
    {query}

    Matches:
    {matches}
    """
    
    FILTER_PROMPT = FILTER_PROMPT.format(query=query, matches=matches)

    filter_messages = [
        {"role": "system", "content": "I have a query and a list of matches obtained from a search based on that query. Please provide a response containing only the matches that are directly relevant to the query. "},
        {"role": "user", "content": {FILTER_PROMPT}},
    ]
    
    filter_response = llama3_8b_ins(llm_tokenizer, llm_model, filter_messages)

    return filter_response

def verify(question, reference, response, llm_tokenizer, llm_model):
    VERIFY_PROMPT = """\
    Question:
    {question}

    Reference Answer:
    {reference}

    Response:
    {response}
    """

    VERIFY_PROMPT = VERIFY_PROMPT.format(question = question, reference=reference, response=response)

    verify_messages = [
        {"role": "system", "content": "When presented with a <Question>, analyze a <Reference Answer> alongside a <Response>. Evaluate the <Response> on a scale from 0 to 100 based on its comparison with the <Reference Answer>. Please provide only the score. "},
        {"role": "user", "content": {VERIFY_PROMPT}},
    ]

    verify_response = llama3_8b_ins(llm_tokenizer, llm_model, verify_messages)
    
    return verify_response


def rag_test_pipeline(DATASET, chunks, stc_tsfm_model, llm_tokenizer, llm_model, write_path):
    len_data = len(DATASET["train"]["question"])
    random_numbers = random.sample(range(0, len_data), 50)

    for i in random_numbers:
        write(write_path, "question: " + DATASET["train"]["question"][i] + "\n\n")
        write(write_path, "answer: " + DATASET["train"]["answer"][i] + "\n\n")
        
        query = DATASET["train"]["question"][i]
        start = time.time()
        matches, scores = search(chunks, query, model = stc_tsfm_model, top_k=10)
        end = time.time()
        execution_time = end - start
        write(write_path, "Retrieved: \n" + str(matches) + "\n" + str(scores)+ "\ntime spent: "+ "{:.2f}".format(execution_time) + "\n\n")
        
        context = matches

        # context, execution_time = filter(query, matches, llm_tokenizer, llm_model)
        # write(write_path, "Filtered context: \n" + str(context)+ "\ntime spent: "+str(execution_time) + "\n\n")


        RAG_PROMPT = """\
        Question:
        {question}

        Context:
        {context}
        """

        RAG_PROMPT = RAG_PROMPT.format(question=query, context=context)

        messages = [
            {"role": "system", "content": "Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'."},
            {"role": "user", "content": {RAG_PROMPT}},
        ]
        write(write_path, str(messages) + "\n\n")

        start = time.time()        
        answer = llama3_8b_ins(llm_tokenizer, llm_model, messages)
        end = time.time()
        execution_time = end - start
        
        write(write_path, "LLAMA3: " + str(answer) + "\ntime spent: "+"{:.2f}".format(execution_time)+ "\n\n")

        #Verify the answer by LLM
        reference = DATASET["train"]["answer"][i]
        start = time.time()        
        verify_response = verify(question = query, reference = reference, response = answer, llm_tokenizer = llm_tokenizer, llm_model = llm_model)
        end = time.time()
        execution_time = end - start
        write(write_path, "Verified as: " + str(verify_response) + ", time spent: "+"{:.2f}".format(execution_time) + "\n\n" + "*"*100 + "\n\n")
        

DATASET = load_dataset("neural-bridge/rag-dataset-1200")


SENTENCE_TSFMS = ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", 'bge-large-en-v1.5','bge-base-en-v1.5', "bge-m3", "thenlper/gte-base", "thenlper/gte-large"]
# SENTENCE_TSFMS = ['BAAI/bge-base-en-v1.5', "BAAI/bge-m3", "thenlper/gte-base", "thenlper/gte-large"]
LLM_ID = "meta-llama/Meta-Llama-3-8B-Instruct"



if __name__ == "__main__":
    # chunks = chunk(DATASET)
    # write("output/rag-dataset-1200_chunks", "CORPUS: \n"+str(chunks)+"\n\n"+"*"*100+"\n\n")
    
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
    llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    )
    with open("output/rag-dataset-1200_chunks.txt", 'r') as file:
            chunks = eval(file.read())
    for i in range(len(SENTENCE_TSFMS)):
        sentence_tsfm_choosen = SENTENCE_TSFMS[i]
        WRITE_PATH = f"output/RAG_test/random50_top10_(llama3-8b-ins)/random50_top10_(llama3-8b-ins)_({sentence_tsfm_choosen.replace('/', '_')}).txt"
        write(WRITE_PATH, "MODELs: \n"+ "SentenceTransformer: "+sentence_tsfm_choosen +"\nLLM: "+LLM_ID+ "\n\n" + "*"*100+"\n\n")
        stc_tsfm_model = SentenceTransformer(sentence_tsfm_choosen)

        rag_test_pipeline(DATASET, chunks, stc_tsfm_model, llm_tokenizer, llm_model, WRITE_PATH)

    