import torch
import time

from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer, util


all_Mini_LM = "all-MiniLM-L6-v2"
bge_large_en = 'BAAI/bge-large-en-v1.5'
bge_m3 = "BAAI/bge-m3"
sfr_embedding_mistral = "Salesforce/SFR-Embedding-Mistral"
gte_base = "thenlper/gte-base"
gte_large = "thenlper/gte-large"

def search(passages:list, query:str, model= SentenceTransformer("all-MiniLM-L6-v2"), top_k = 5):
    queries_embeddings = model.encode(query)
    query_embeddings = torch.FloatTensor(queries_embeddings)
    passages_embeddings = model.encode(passages)
    hits = util.semantic_search(query_embeddings, passages_embeddings, top_k=top_k)
    matches = [passages[hits[0][i]['corpus_id']] for i in range(len(hits[0]))]
    scores = [[hits[0][i]['score']] for i in range(len(hits[0]))]
    return matches, scores

if __name__ == "__main__":

    passages = ["How do I get a replacement Medicare card?",
        "What is the monthly premium for Medicare Part B?",
        "How do I terminate my Medicare Part B (medical insurance)?",
        "How do I sign up for Medicare?",
        "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
        "How do I sign up for Medicare Part B if I already have Part A?",
        "What are Medicare late enrollment penalties?",
        "What is Medicare and who can get it?",
        "How can I get help with my Medicare Part A and Part B premiums?",
        "What are the different parts of Medicare?",
        "Will my Medicare premiums be higher because of my higher income?",
        "What is TRICARE ?",
        "Should I sign up for Medicare Part B if I have Veterans' Benefits?"]
    
    query = "How can Medicare help me?"

    start = time.time()

    model = SentenceTransformer(gte_large)
    matches, scores = search(passages, query, model = model)

    print(matches)
    print(scores)

    end = time.time()
    print()
    print(f'elapsed time: {end-start:.2f}s')