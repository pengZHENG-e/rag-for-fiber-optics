from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import transformers
import torch


def init_llama3_8b_ins():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return tokenizer, model


def llama3_8b_ins(tokenizer, model, messages, max_new_tokens = 512, do_sample = True, temp = 0.6, top_p= 0.9):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=do_sample,
        temperature=temp,
        top_p=top_p,
    )
    response = outputs[0][input_ids.shape[-1]:]

    output = tokenizer.decode(response, skip_special_tokens=True)
    return output


def llama3_8b(model_id, message):
    pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
    )

    return pipeline(message)


def init_phi3_mini_4k_ins():
    model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    return tokenizer, model


def phi3_mini_4k_ins(tokenizer, model, messages):
    pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.5,
        "do_sample": True,
    }

    output = pipe(messages, **generation_args)[0]['generated_text']

    return output


def init_Llama3_ChatQA():
    model_id = "nvidia/Llama3-ChatQA-1.5-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    
    return tokenizer, model


def Llama3_ChatQA(tokenizer, model, formatted_input):

    # messages= [
    #     {"role": "user", "content": "what is the percentage change of the net income from Q4 FY23 to Q4 FY24?"}
    # ]

    # document = """NVIDIA (NASDAQ: NVDA) today reported revenue for the fourth quarter ended January 28, 2024, of $22.1 billion, up 22% from the previous quarter and up 265% from a year ago.\nFor the quarter, GAAP earnings per diluted share was $4.93, up 33% from the previous quarter and up 765% from a year ago. Non-GAAP earnings per diluted share was $5.16, up 28% from the previous quarter and up 486% from a year ago.\nQ4 Fiscal 2024 Summary\nGAAP\n| $ in millions, except earnings per share | Q4 FY24 | Q3 FY24 | Q4 FY23 | Q/Q | Y/Y |\n| Revenue | $22,103 | $18,120 | $6,051 | Up 22% | Up 265% |\n| Gross margin | 76.0% | 74.0% | 63.3% | Up 2.0 pts | Up 12.7 pts |\n| Operating expenses | $3,176 | $2,983 | $2,576 | Up 6% | Up 23% |\n| Operating income | $13,615 | $10,417 | $1,257 | Up 31% | Up 983% |\n| Net income | $12,285 | $9,243 | $1,414 | Up 33% | Up 769% |\n| Diluted earnings per share | $4.93 | $3.71 | $0.57 | Up 33% | Up 765% |"""

    # formatted_input = get_formatted_input(messages, document)
    tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=128, eos_token_id=terminators)

    response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def get_ins_messages(sys, user):
    messages = [
    {"role": "system", "content": {sys}},
    {"role": "user", "content": {user}},
    ]
    return messages

def main():
    tokenizer, model = init_llama3_8b_ins()
    sys = "You are cool post punk band singer!"
    user = "Who are you?"
    messages = get_ins_messages(sys, user)

    answer = llama3_8b_ins(tokenizer, model, messages)
    print(answer)


if __name__ == "__main__":
    main()