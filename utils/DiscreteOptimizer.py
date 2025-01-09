import random
import torch
import copy


class DiscreteOptimizer:
    def __init__(self, batch, original_embedding_output, neighbours, tokenizer, embedding_layer, valid_length, device,
                 beam_width):
        self.batch = batch
        self.batch_tokens = batch["input_ids"].tolist()
        self.original_embedding_output = original_embedding_output
        self.neighbours = neighbours
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
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

    def objective_function(self, solution):
        shadow_embedding_output = self.embedding_layer(torch.tensor(solution).to(self.device))
        shadow_embedding_output_mean = torch.mean(shadow_embedding_output, dim=1)
        original_embedding_output_mean = torch.mean(self.original_embedding_output, dim=1)
        loss = 1 - torch.nn.functional.cosine_similarity(original_embedding_output_mean, shadow_embedding_output_mean,
                                                         dim=1)
        # embedding_cos_distance = []
        # for i in range(len(self.original_embedding_output)):
        #     for j in range(1, self.valid_length - 1):
        #         original_token_embedding = self.original_embedding_output[i][j]
        #         shadow_token_embedding = shadow_embedding_output[i][j]
        #         embedding_cos_distance.append(1 - torch.nn.functional.cosine_similarity(original_token_embedding,
        #                                                                                 shadow_token_embedding, dim=0))
        # loss = torch.mean(torch.tensor(embedding_cos_distance)).item()
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
