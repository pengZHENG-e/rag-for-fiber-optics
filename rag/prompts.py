def get_formatted_input(messages, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."
    
    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + context + "\n\n" + conversation
    
    return formatted_input


def get_formatted_input_HyDE(messages):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."
    
    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + conversation
    
    return formatted_input


def get_formatted_input_with_memory(messages, contxt, memory):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."
    
    his = """Relevant pieces of previous conversation:
    {history}
    (You do not need to use these pieces of information if not relevant)"""
    his = his.format(history = memory)

    context = """Relevant context:
    {cont}
    (You do not need to use these pieces of information if not relevant)"""
    context = context.format(cont = contxt)

    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + his + "\n\n" + context + "\n\n" + conversation
    
    return formatted_input


generate_questions_V2 = """\
        Given the context information and not prior knowledge.
        generate several(around 5) only questions and their answers only about the value of DTS technology for the final user based on the below query. 
        Also remember just response the questions in the format: "{"Questions": ["content of q1", "content of q2",...],"Answers: ["answer of q1", "answer of q2",...]}". 
        Don't response things including like "Here are 5 questions based on the context information:". Just the "{"Questions": ["content of q1", "content of q2",...],"Answers: ["answer of q1", "answer of q2",...]}". 

        You are a Fiber Optic Teacher/ Professor. Your task is to setup \
        several(around 5) questions and their answers only about the value of DTS technology for the final user for an upcoming \
        quiz/examination. The questions should be diverse in nature \
        across the document. Restrict the questions to the \
        context information provided."""


generate_questions_V1 = """\
        Given the context information and not prior knowledge.
        generate 5 only questions based on the below query. 
        Also remember just response the questions in the format: "["content of q1", "content of q2",...]". 
        Don't response things including like "Here are 5 questions based on the context information:". Just the "["content of q1", "content of q2",...]". 

        You are a Teacher/ Professor. Your task is to setup \
        5 questions for an upcoming \
        quiz/examination. The questions should be diverse in nature \
        across the document. Restrict the questions to the \
        context information provided."""