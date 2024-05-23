import nltk # Using NLTK 
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import spacy
nlp = spacy.load("en_core_web_sm")

def sentence_chunking_nltk(text):
    sentences = sent_tokenize(text)
    return sentences


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

def chunk_by_sentence(text):
    sentences = text.split("\\n")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    return sentences

def sentence_chunking_spacy(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences



if __name__ == "__main__":
    with open("data/text_example_1.txt", 'r') as file:
        text = file.read()

    chunks = sentence_chunking_spacy(text)

    for i, chunk in enumerate(chunks):  
        print(f"Chunk {i+1}:\n{'-'*10}\n{chunk}\n")
        print(f"{'-'*100}\n{'-'*100}\n")


