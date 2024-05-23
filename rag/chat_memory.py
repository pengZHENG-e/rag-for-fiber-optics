#https://nanonets.com/blog/langchain/#module-v-memory

from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory


class Memory:
    def __init__(self, k=5):
        self.memory = ConversationBufferWindowMemory(k=k)

    def save(self, input_str, output_str):
        self.memory.save_context({"input": input_str}, {"output": output_str})

    def load(self):
        return self.memory.load_memory_variables({})["history"]

_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""

if __name__ == "__main__":
    # memory = ConversationBufferMemory()
   
    memory = Memory(k = 6)
    memory.save("hello", "hi")
    memory.save("what's up", "good")
    his = memory.load()
    print(his)

