from sentence_transformers import SentenceTransformer, losses, InputExample
from datasets import load_dataset
from torch.utils.data import DataLoader


model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# dataset_id = "embedding-data/QQP_triplets"
# dataset = load_dataset(dataset_id)
# print(f"Examples look like this: {dataset['train']['set'][:10]}")

dataset_id = "embedding-data/sentence-compression"
dataset = load_dataset(dataset_id)
print("\n\n\n")
print(f"Examples look like this: {dataset['train']['set'][0]}")

# train_examples = []
# train_data = dataset['train']['set']
# n_examples = dataset['train'].num_rows

# # example = train_data[0]
# # print(example[0])
# # print(example[1])

# for i in range(n_examples):
#   example = train_data[i]
#   train_examples.append(InputExample(texts=[example[0], example[1]]))

# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
# train_loss = losses.MultipleNegativesRankingLoss(model=model)
# num_epochs = 10
# warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data

# model.fit(train_objectives=[(train_dataloader, train_loss)],
#           epochs=num_epochs,
#           warmup_steps=warmup_steps)


#####################################################################################

