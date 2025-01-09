import torch
import copy
import spacy


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


class MyDummyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["train_transformed_embeddings"])


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # len(s1) >= len(s2)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_response(client, conversation, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": conversation}
        ]
    )
    return response.choices[0].message.content


# Define a custom embedding module
class GPT2CombinedEmbeddings(torch.nn.Module):
    def __init__(self, gpt2_model):
        super(GPT2CombinedEmbeddings, self).__init__()
        self.word_embeddings = gpt2_model.wte  # Word embeddings
        self.position_embeddings = gpt2_model.wpe  # Positional embeddings

    def forward(self, input_ids):
        # Get word embeddings for the input IDs
        word_embeds = self.word_embeddings(input_ids)

        # Generate position IDs and get positional embeddings
        position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_embeds = self.position_embeddings(position_ids)

        # Combine word and positional embeddings
        combined_embeds = word_embeds + position_embeds
        return combined_embeds


def get_topk_neighbours(topk_neighbours, overlap, topk):
    new_topk_neighbours = copy.deepcopy(topk_neighbours)
    #
    for i in range(len(topk_neighbours)):
        token = i
        neighbours = topk_neighbours[i]
        for neighbour in neighbours:
            neighbour_neighbours = topk_neighbours[neighbour]
            # Calculate the intersection of the two sets
            intersection = set(neighbours).intersection(neighbour_neighbours)
            if len(intersection) / topk > overlap:
                new_topk_neighbours[i].remove(neighbour)

    for i in range(len(new_topk_neighbours)):
        if not new_topk_neighbours[i]:
            new_topk_neighbours[i] = topk_neighbours[i]

    new_new_topk_neighbours = copy.deepcopy(new_topk_neighbours)

    for i in range(len(new_topk_neighbours)):
        token = i
        neighbours = new_topk_neighbours[i]
        if len(neighbours) == 1:
            continue
        for neighbour in neighbours:
            if token in new_topk_neighbours[neighbour]:
                if neighbour in new_new_topk_neighbours[token]:
                    new_new_topk_neighbours[token].remove(neighbour)
                if token in new_new_topk_neighbours[neighbour]:
                    new_new_topk_neighbours[neighbour].remove(token)

    for i in range(len(new_new_topk_neighbours)):
        if not new_new_topk_neighbours[i]:
            new_new_topk_neighbours[i] = topk_neighbours[i]

    return new_new_topk_neighbours


def get_topk_neighbours_lemma(topk_neighbours, overlap, topk, tokenizer):
    new_topk_neighbours = copy.deepcopy(topk_neighbours)
    lemma_cache = {}
    spacy_nlp = spacy.load("en_core_web_sm")
    #
    for i in range(len(topk_neighbours)):
        if i % 100 == 0:
            print("Filtering token", i + 1, "out of", len(topk_neighbours), "based on overlap")

        token = i
        doc = spacy_nlp(tokenizer.decode(token).strip())
        if len(doc) == 0:
            lemma_cache[token] = str.lower(tokenizer.decode(token).strip())
        else:
            lemma_cache[token] = str.lower(doc[0].lemma_)

        neighbours = topk_neighbours[i]
        for neighbour in neighbours:
            neighbour_neighbours = topk_neighbours[neighbour]
            # Calculate the intersection of the two sets
            intersection = set(neighbours).intersection(neighbour_neighbours)
            if len(intersection) / topk > overlap:
                new_topk_neighbours[i].remove(neighbour)

    for i in range(len(new_topk_neighbours)):
        if not new_topk_neighbours[i]:
            new_topk_neighbours[i] = topk_neighbours[i]

    new_new_topk_neighbours = copy.deepcopy(new_topk_neighbours)

    for i in range(len(new_topk_neighbours)):
        if i % 100 == 0:
            print("Filtering token", i + 1, "out of", len(topk_neighbours), "based on overlapping neighbours")
        token = i
        neighbours = new_topk_neighbours[i]
        if len(neighbours) == 1:
            continue
        for neighbour in neighbours:
            if token in new_topk_neighbours[neighbour]:
                if neighbour in new_new_topk_neighbours[token]:
                    new_new_topk_neighbours[token].remove(neighbour)
                if token in new_new_topk_neighbours[neighbour]:
                    new_new_topk_neighbours[neighbour].remove(token)

    for i in range(len(new_new_topk_neighbours)):
        if not new_new_topk_neighbours[i]:
            new_new_topk_neighbours[i] = topk_neighbours[i]


    new_new_new_topk_neighbours = copy.deepcopy(new_new_topk_neighbours)

    for i in range(len(new_new_topk_neighbours)):
        if i % 100 == 0:
            print("Filtering token", i + 1, "out of", len(topk_neighbours), "based on lemma")
        token = i
        neighbours = new_new_topk_neighbours[i]
        if len(neighbours) == 1:
            continue
        for neighbour in neighbours:
            if lemma_cache[token] == lemma_cache[neighbour]:
                if neighbour in new_new_new_topk_neighbours[token]:
                    new_new_new_topk_neighbours[token].remove(neighbour)
                if token in new_new_new_topk_neighbours[neighbour]:
                    new_new_new_topk_neighbours[neighbour].remove(token)
            else:
                continue

    for i in range(len(new_new_new_topk_neighbours)):
        if not new_new_new_topk_neighbours[i]:
            new_new_new_topk_neighbours[i] = topk_neighbours[i]

    return new_new_new_topk_neighbours