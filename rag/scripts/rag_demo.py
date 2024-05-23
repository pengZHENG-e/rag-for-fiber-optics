import sys
sys.path.append(".")

from rag.chunk.semantic_chunk import chunk
from rag.scripts.embedders import search
from rag.LLMs import llama3_8b_ins
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_text(file_path):
    """
    Load text from a file.
    """
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def chunk_text(text, model_name):
    """
    Chunk the given text using the specified model.
    """
    chunks = chunk(text, model_name)
    return chunks

def search_chunks(chunks, question, model):
    """
    Search the chunks for relevant matches to the given question.
    """
    matches, scores = search(chunks, question, model)
    return matches, scores

def format_rag_prompt(question, context):
    """
    Format the RAG prompt with the given question and context.
    """
    rag_prompt = """\
Question: {question}
Context: {context}
"""
    rag_prompt = rag_prompt.format(question=question, context=context)
    return rag_prompt

def get_answer(tokenizer, model, rag_prompt):
    """
    Get the answer from the LLM model using the RAG prompt.
    """
    messages = [
        {"role": "system", "content": "Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'."},
        {"role": "user", "content": rag_prompt},
    ]
    answer = llama3_8b_ins(tokenizer, model, messages)
    return answer

def main():
    file_path = 'data/text_example.txt'
    article = load_text(file_path)
    model_name = "BlueOrangeDigital/distilbert-cross-segment-document-chunking"
    chunks = chunk_text(article, model_name)
    question = "Is there any people died?"
    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
    matches, _ = search_chunks(chunks, question, sentence_transformer_model)
    context = matches[0]
    rag_prompt = format_rag_prompt(question, context)
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    answer = get_answer(tokenizer, model, rag_prompt)
    print(answer)

if __name__ == "__main__":
    main()