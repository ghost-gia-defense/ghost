import random
import torch
import copy


class DiscreteOptimizerFullParallel:
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
        tokens = self.batch_tokens[0]
        for token in tokens:
            if token in self.tokenizer.all_special_ids:
                solution.append(token)
            else:
                solution.append(random.choice(self.neighbours[token]))
        self.beams.append(copy.deepcopy(solution))
        self.beam_objs.extend(self.objective_function(solution))

    def objective_function(self, solutions):
        solutions = torch.tensor(solutions).to(self.device)
        # Make sure solutions is of shape (batch_size, sequence_length)
        if len(solutions.size()) == 1:
            solutions = solutions.unsqueeze(0)
        repeated_attention_mask = self.batch["attention_mask"].repeat(solutions.size(0), 1).to(self.device)
        with torch.no_grad():
            shadow_outputs = self.model(input_ids=solutions,
                                        attention_mask=repeated_attention_mask,
                                        output_hidden_states=True)
        shadow_hidden_states = shadow_outputs.hidden_states
        # Calculate the MSE between the original hidden states and the shadow hidden states
        losses = []
        for i in range(solutions.size(0)):
            loss = 0
            for j in range(len(self.original_hidden_states)):
                loss += torch.nn.functional.mse_loss(self.original_hidden_states[j], shadow_hidden_states[j][i])
            losses.append(loss.item())

        return losses

    def optimize(self):
        i = 0
        current_beams = copy.deepcopy(self.beams)
        current_beam_objs = copy.deepcopy(self.beam_objs)

        while True:
            print("Optimization iter", i + 1)
            print("Current best loss:", current_beam_objs[0])
            for j in range(len(current_beams)):
                current_beam = copy.deepcopy(current_beams[j])
                for l in range(1, self.valid_length - 1):
                    neighbours = self.neighbours[self.batch_tokens[0][l]]
                    temp_solutions = []
                    for neighbour in neighbours:
                        temp_solution = copy.deepcopy(current_beam)
                        temp_solution[l] = neighbour
                        temp_solutions.append(copy.deepcopy(temp_solution))
                    temp_solution_objs = self.objective_function(temp_solutions)
                    current_beams.extend(copy.deepcopy(temp_solutions))
                    current_beam_objs.extend(copy.deepcopy(temp_solution_objs))
                    #
                    # for neighbour in self.neighbours[self.batch_tokens[k][l]]:
                    #     temp_solution = copy.deepcopy(current_beam)
                    #     temp_solution[k][l] = neighbour
                    #     current_beams.append(copy.deepcopy(temp_solution))
                    #     current_beam_objs.append(self.objective_function(temp_solution))
                    # Only keep the best beams based on the beam width
                    sorted_indices = torch.argsort(torch.tensor(current_beam_objs))
                    current_beams = [current_beams[idx] for idx in sorted_indices[:self.beam_width]]
                    current_beam_objs = [current_beam_objs[idx] for idx in sorted_indices[:self.beam_width]]

            if current_beam_objs[0] == self.beam_objs[0]:
                print("Converged")
                break
            elif self.beam_objs[0] - current_beam_objs[0] < 0.1:
                print("Early stopping")
                self.beams = copy.deepcopy(current_beams)
                self.beam_objs = copy.deepcopy(current_beam_objs)
                break
            else:
                self.beams = copy.deepcopy(current_beams)
                self.beam_objs = copy.deepcopy(current_beam_objs)
            i += 1

        return self.beams[0], self.beam_objs[0]
