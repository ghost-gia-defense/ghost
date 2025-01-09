from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from utils import *
import json
import random
from argparse import ArgumentParser
import os

parser = ArgumentParser("Transform data using discrete optimization")
parser.add_argument("--data", type=str, default="sst2", help="The dataset to use")
parser.add_argument("--device", type=str, default="cuda:0", help="The device to use")
parser.add_argument("--num_of_samples", type=int, default=50, help="The number of samples to use")
parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-small", help="The model to use")
parser.add_argument("--beam_width", type=int, default=1, help="The beam width to use")
parser.add_argument("--overlap", type=float, default=0.1, help="The overlap between neighbours")
parser.add_argument("--topk", type=int, default=70, help="The number of neighbours to use")
parser.add_argument("--recover_batch", type=int, default=0, help="The batch to recover from")
args = parser.parse_args()

random.seed(42)
torch.manual_seed(42)

data = args.data
device = args.device
num_of_samples = args.num_of_samples
beam_width = args.beam_width
model_name = args.model_name
topk = args.topk
overlap = args.overlap
recover_batch = args.recover_batch

print("Model:", model_name)

directory = f"../data/{model_name.split('/')[-1]}"
if not os.path.exists(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

model = AutoModel.from_pretrained(model_name)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
if data == "cola" or data == "sst2":
    dataset = load_dataset("glue", data)["train"]
    sentences = dataset["sentence"]
    labels = dataset["label"]
elif data == "rotten_tomatoes":
    dataset = load_dataset(data)["train"]
    sentences = dataset["text"]
    labels = dataset["label"]
elif data == "tweeter":
    dataset = load_dataset("SetFit/tweet_sentiment_extraction")["train"]
    sentences = dataset["text"]
    labels = dataset["label"]
elif data == "yahoo":
    dataset = load_dataset("yahoo_answers_topics")["train"]
    sentences = dataset["question_title"]
    labels = dataset["topic"]

# Get same number of samples from each class
indices = []
unique_labels = list(set(labels))
for i in range(len(unique_labels)):
    unique_labels[i] = i

for label in unique_labels:
    indices.append([j for j in range(len(labels)) if labels[j] == label])
shuffled_indices = []
for i in range(len(indices)):
    shuffled_indices += random.choices(indices[i], k=int(num_of_samples / len(set(labels))))
sentences = [sentences[i] for i in shuffled_indices]
labels = [labels[i] for i in shuffled_indices]

token_embeddings = model.get_input_embeddings().weight

# Compare all token embeddings similarity with each other in a vectorized way
# Normalize the embeddings
normalized_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=1)

# Compute cosine similarities
cos_sims_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
#
# Get each token's top 1000 most similar tokens and exclude itself
_, topk_indices = torch.topk(cos_sims_matrix, topk + 1, dim=1)
topk_neighbours = topk_indices.tolist()
for i in range(len(topk_neighbours)):
    topk_neighbours[i] = topk_neighbours[i][1:]

topk_neighbours = get_topk_neighbours_lemma(topk_neighbours, overlap, topk, tokenizer)

#
# Tokenize the sentences
tokenized_sentences = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
dataset = MyDataset(tokenized_sentences)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
#
#
#
if recover_batch > 0:
    previous_data = json.load(open(
        f"../data/{model_name.split('/')[-1]}/{data}_top_{topk}_beam_{beam_width}_overlap_{overlap}_discrete_transformed_data.json",
        "r"))
    transformed_data = previous_data["transformed_sentences"]
else:
    transformed_data = []

i = 0
for batch in dataloader:
    if i < recover_batch:
        i += 1
        continue
    print("Processing batch ", i + 1, "out of", len(dataloader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    valid_length = torch.sum(attention_mask, dim=1)
    with torch.no_grad():
        original_output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        original_hidden_states = original_output.hidden_states

    optimizer = DiscreteOptimizerFullParallel(batch=batch, original_hidden_states=original_hidden_states,
                                      neighbours=topk_neighbours, tokenizer=tokenizer,
                                      model=model, valid_length=valid_length, device=device,
                                      beam_width=beam_width)
    optimizer.initialize()
    shadow_sentence, loss = optimizer.optimize()
    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    print(tokenizer.decode(shadow_sentence, skip_special_tokens=True))
    print("Loss:", loss)
    transformed_data.append(tokenizer.decode(shadow_sentence, skip_special_tokens=True))
    i += 1

    with open(
            f"../data/{model_name.split('/')[-1]}/{data}_top_{topk}_beam_{beam_width}_overlap_{overlap}_discrete_transformed_data.json",
            "w") as f:
        json.dump({"transformed_sentences": transformed_data, "labels": labels, "original_sentences": sentences}, f)
