from langchain_community.document_loaders import PyPDFLoader

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append(".")
from rag.LLMs import llama3_8b_ins
import rag.prompts as prompts

class QuestionGenerator:
    def __init__(self, model_id):
        self.tokenizer, self.model = self.load_model(model_id)

    def load_model(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        return tokenizer, model

    def generate_questions(self, chunks):
        system_content = prompts.generate_questions_V2
        
        data = []
        for chunk in chunks:
            user_content = f"""\
            Context information is below.
            
            ---------------------
            {chunk}
            ---------------------
            """
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            pairs = llama3_8b_ins(self.tokenizer, self.model, messages)
            data.append({"Context": chunk, "QA-pairs": pairs})

        return pd.DataFrame(data)

if __name__ == "__main__":
    file_path = "data/Neon Op manual_5030412_01.pdf"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    output_file = "output/generated_QAs/by_pages/Neon-Op-manual_by_pages.csv"
    
    loader = PyPDFLoader(file_path, extract_images=False)
    chunks = [doc.page_content for doc in loader.load()]
    
    question_generator = QuestionGenerator(model_id)
    df = question_generator.generate_questions(chunks)
    df.to_csv(output_file, index=True)