import random
import torch
import copy
import spacy


class DiscreteOptimizerFullLemma:
    def __init__(self, batch, original_hidden_states, neighbours, tokenizer, model, valid_length, device,
                 beam_width):
        self.batch = batch
        self.batch_tokens = batch["input_ids"].tolist()
        self.original_hidden_states = original_hidden_states
        self.neighbours = neighbours
        self.tokenizer = tokenizer
        self.model = model
        self.valid_length = valid_length
        self.device = device
        self.beams = []
        self.beam_objs = []
        self.beam_width = beam_width

    def initialize(self):
        solution = []
        for i in range(len(self.batch_tokens)):
            tokens = self.batch_tokens[i]
            dummy_tokens = []
            for token in tokens:
                if token in self.tokenizer.all_special_ids:
                    dummy_tokens.append(token)
                else:
                    dummy_tokens.append(random.choice(self.neighbours[token]))
            solution.append(dummy_tokens)
        self.beams.append(copy.deepcopy(solution))
        self.beam_objs.append(self.objective_function(solution))

        # Eliminate the neighbours that have the same lemma with the original token
        spacy_nlp = spacy.load("en_core_web_sm")
        neighbours = copy.deepcopy(self.neighbours)
        for i in range(len(self.batch_tokens)):
            tokens = self.batch_tokens[i]
            for token in tokens:
                token_neighbours = neighbours[token]
                if len(token_neighbours) == 1:
                    continue
                for neighbour in token_neighbours:
                    doc = spacy_nlp(
                        self.tokenizer.decode(token).strip() + " " + self.tokenizer.decode(neighbour).strip())
                    if len(doc) < 2:
                        continue
                    elif str.lower(doc[0].lemma_) == str.lower(doc[1].lemma_):
                        if neighbour in self.neighbours[token]:
                            self.neighbours[token].remove(neighbour)
                        if token in self.neighbours[neighbour]:
                            self.neighbours[neighbour].remove(token)
                    else:
                        continue

        for i in range(len(self.neighbours)):
            if not self.neighbours[i]:
                self.neighbours[i] = neighbours[i]


    def objective_function(self, solution):
        with torch.no_grad():
            shadow_output = self.model(input_ids=torch.tensor(solution).to(self.device),
                                       attention_mask=self.batch["attention_mask"].to(self.device),
                                       output_hidden_states=True)
        shadow_hidden_states = shadow_output.hidden_states
        # Calculate the MSE between the original hidden states and the shadow hidden states
        loss = 0
        for i in range(len(self.original_hidden_states)):
            loss += torch.nn.functional.mse_loss(self.original_hidden_states[i], shadow_hidden_states[i])

        return loss.item()

    def optimize(self):
        i = 0
        current_beams = copy.deepcopy(self.beams)
        current_beam_objs = copy.deepcopy(self.beam_objs)

        while True:
            print("Optimization iter", i + 1)
            print("Current best loss:", current_beam_objs[0])
            for j in range(len(current_beams)):
                current_beam = copy.deepcopy(current_beams[j])
                for k in range(len(current_beam)):
                    for l in range(1, self.valid_length - 1):
                        for neighbour in self.neighbours[self.batch_tokens[k][l]]:
                            temp_solution = copy.deepcopy(current_beam)
                            temp_solution[k][l] = neighbour
                            current_beams.append(copy.deepcopy(temp_solution))
                            current_beam_objs.append(self.objective_function(temp_solution))
                        # Only keep the best beams based on the beam width
                        sorted_indices = torch.argsort(torch.tensor(current_beam_objs))
                        current_beams = [current_beams[idx] for idx in sorted_indices[:self.beam_width]]
                        current_beam_objs = [current_beam_objs[idx] for idx in sorted_indices[:self.beam_width]]

            if current_beam_objs[0] == self.beam_objs[0]:
                break
            else:
                self.beams = copy.deepcopy(current_beams)
                self.beam_objs = copy.deepcopy(current_beam_objs)
            i += 1

        return self.beams[0], self.beam_objs[0]
