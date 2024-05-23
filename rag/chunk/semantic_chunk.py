import sys
sys.path.append(".")
from rag.scripts.embedders import search

from transformers import (
   DistilBertTokenizer,
   AutoModelForSequenceClassification,
   TextClassificationPipeline
)
from sentence_transformers import SentenceTransformer


def right_truncate_sentence(sentence, tokenizer, max_len):
   tokenized = tokenizer.encode(sentence)[1:-1]
   if len(tokenized) > max_len:
       print("cut")
   return tokenizer.decode(tokenized[:max_len])

def left_truncate_sentence(sentence, tokenizer, max_len):
   tokenized = tokenizer.encode(sentence)[1:-1]
   if len(tokenized) > max_len:
       print("cut")
   return tokenizer.decode(tokenized[-max_len:])

def bucket_pair(left_sentence, right_sentence, tokenizer, max_len):
   return left_truncate_sentence(left_sentence, tokenizer, max_len) + " [SEP] " + \
       right_truncate_sentence(right_sentence, tokenizer, max_len)


def chunk(article, model_name):
   MAX_LEN = 255
   ordered_sentences = article.split('\n')[:-1]
   tokenizer = DistilBertTokenizer.from_pretrained(model_name)
   pairs = [
      bucket_pair(ordered_sentences[i], ordered_sentences[i+1], tokenizer, MAX_LEN)
      for i in range(0, len(ordered_sentences) - 1)
   ]
   id2label = {0: "DIFFERENT", 1: "SAME"}
   label2id = {"DIFFERENT": 0, "SAME": 1}

   model = AutoModelForSequenceClassification.from_pretrained(
      model_name,
      num_labels=2,
      id2label=id2label,
      label2id=label2id
   )

   pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

   predictions = pipe(pairs)

   n = len(ordered_sentences)
   chunks_breaks = [
   i+1
   for i, pred in enumerate(predictions)
   if pred["label"] != "DIFFERENT"
   ]

   chunks = [
      "\n".join(ordered_sentences[i:j])
      for i, j in zip([0] + chunks_breaks, chunks_breaks + [n])
   ]

   return chunks

if __name__ == "__main__":
   
   with open('data/text_example.txt', 'r') as file:
      article = file.read()

   model_name = "BlueOrangeDigital/distilbert-cross-segment-document-chunking"
   chunks = chunk(article, model_name)

   for i, chunk in enumerate(chunks):  
      print(f"Chunk {i+1}:\n{'-'*10}\n{chunk}\n")
      print(f"{'-'*100}\n{'-'*100}\n")


   matches, scores = search(chunks, "Is there any people died?", SentenceTransformer("all-MiniLM-L6-v2"))

   print(matches)
   print(scores)
