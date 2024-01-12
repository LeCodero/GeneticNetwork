import torch
import random
from neural_networks.model import NeuralNetwork

def classic_averaging_crossover(parent1, parent2, generation, parent_ids):
    child = NeuralNetwork(parent=parent1, generation=generation, parent_ids=parent_ids)
    child_state_dict = {}

    parent1_state_dict = parent1.model.state_dict()
    parent2_state_dict = parent2.model.state_dict()

    with torch.no_grad():
        for key in parent1_state_dict:
            child_state_dict[key] = 0.5 * (parent1_state_dict[key] + parent2_state_dict[key])

    child.model.load_state_dict(child_state_dict)
    return child

def swa_crossover(parent1, parent2, generation, parent_ids, swa_rate=0.1):
    child = NeuralNetwork(parent=parent1, generation=generation, parent_ids=parent_ids)
    child_state_dict = parent1.model.state_dict()

    parent2_state_dict = parent2.model.state_dict()

    with torch.no_grad():
        for key in child_state_dict:
            child_state_dict[key] = (1 - swa_rate) * child_state_dict[key] + swa_rate * parent2_state_dict[key]

    child.model.load_state_dict(child_state_dict)
    return child

def weighted_averaging_crossover(parent1, parent2, generation, parent_ids, weight=0.7):
    child = NeuralNetwork(parent=parent1, generation=generation, parent_ids=parent_ids)
    child_state_dict = {}

    parent1_state_dict = parent1.model.state_dict()
    parent2_state_dict = parent2.model.state_dict()

    with torch.no_grad():
        for key in parent1_state_dict:
            child_state_dict[key] = weight * parent1_state_dict[key] + (1 - weight) * parent2_state_dict[key]

    child.model.load_state_dict(child_state_dict)
    return child

def random_choice_crossover(parent1, parent2, generation, parent_ids):
    methods = [classic_averaging_crossover, swa_crossover, weighted_averaging_crossover]
    chosen_method = random.choice(methods)
    return chosen_method(parent1, parent2, generation, parent_ids)