from .fitness import evaluate_fitness
from .population import initialize_population, reproduce
from .crossover import *
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy

class GeneticAlgorithmEngine:
    def __init__(self):
        self.population = []
        self.best_model = self.selection_method = self.crossover_function = None
        self.epochs = self.early_stopping_rounds = self.no_improvement_count = self.current_generation = 0
        self.best_fitness = float('inf')
        self.device = self.dynamic_population = self.degressive_epochs = self.metrics_function = None
        self.genealogy_info = [{}]
        self.best_model_each_generation = {}
        self.initial_population_size = 0

    def init(self, model_architecture, loss_function, metrics_function=None, dynamic_population=False, optimizer_func=optim.Adam, learning_rate=0.001, selection_method='elitism', population_size=10, crossover_function=swa_crossover, epochs=1, early_stopping_rounds=None, degressive_epochs=False, device='cpu'):
        if population_size < 10:
            raise ValueError('population_size must be a value of minimum 10.')
        self.device = torch.device(device)
        self.initial_population_size = population_size
        self.population, self.genealogy_info = initialize_population(population_size, model_architecture, optimizer_func, learning_rate, loss_function, metrics_function, self.device)
        self.selection_method = selection_method
        self.crossover_function = crossover_function
        self.early_stopping_rounds = early_stopping_rounds
        self.epochs = epochs
        self.dynamic_population = dynamic_population
        self.degressive_epochs = degressive_epochs
        self.metrics_function = metrics_function

    def train(self, train_loader, test_loader):
        found_better_model = False
        for model in tqdm(self.population, desc=f"Training generation {self.current_generation + 1}"):
            model.fitness, model.accuracy = evaluate_fitness(model, train_loader, test_loader, round(self.epochs))
            if model.fitness < self.best_fitness:
                self.best_fitness = model.fitness
                self.best_model = deepcopy(model)
                found_better_model = True
                self.best_model_each_generation[self.current_generation] = self.best_model.id
        self.epochs = max(1, self.epochs * 0.8) if self.degressive_epochs and self.epochs != 1 else self.epochs
        self.no_improvement_count = 0 if found_better_model else self.no_improvement_count + 1
        if self.early_stopping_rounds and self.no_improvement_count >= self.early_stopping_rounds:
            return True
        self.current_generation += 1
        children, self.genealogy_info = reproduce(self.population, self.crossover_function, self.initial_population_size, self.selection_method, self.current_generation, self.genealogy_info, self.dynamic_population)
        self.population = children
        return False


    def get_best_model(self):
        return self.best_model

    def find_ancestors(self, model_id, gen_index):
        ancestors = set()
        if gen_index < 0:
            return ancestors

        model_info = self.genealogy_info[gen_index].get(model_id)
        if model_info:
            parents = model_info['parents']
            for parent_id in parents:
                ancestors.add((parent_id, gen_index - 1))
                ancestors |= self.find_ancestors(parent_id, gen_index - 1)
        
        return ancestors

    def create_node(self, G, model_info, alias, alias_map, displayed_models_this_gen, gen_index, model_id, edge_color_map):
        G.add_node(alias)
        displayed_models_this_gen.append(alias)

        if gen_index > 0:
            for parent_id in model_info['parents']:
                parent_alias = alias_map.get((parent_id, gen_index - 1))
                if parent_alias:
                    G.add_edge(parent_alias, alias)
                    edge_color_map.append('black')
        return G, edge_color_map

    def generate_genealogy_tree(self):
        G = nx.DiGraph()
        pos = {}
        color_map = []
        edge_color_map = []
        alias_map = {}
        survival_count = {}
        base_alias_map = {}
        node_sizes = []

        last_gen_index = max(self.best_model_each_generation)
        if last_gen_index > 25:
            print('Not showing genealogy tree: Too many models to show.')
            return
        self.genealogy_info = self.genealogy_info[:last_gen_index+1]
        best_model_id = self.best_model_each_generation.get(last_gen_index)
        all_ancestors = self.find_ancestors(best_model_id, last_gen_index)

        current_alias = 1
        for gen_index, gen in enumerate(self.genealogy_info):
            displayed_models_this_gen = []

            for model_id, model_info in gen.items():
                if (model_id, gen_index) not in all_ancestors and model_id != best_model_id:
                    continue
                
                base_alias = base_alias_map.get(model_id, str(current_alias))
                base_alias_map[model_id] = base_alias

                survival_count[model_id] = survival_count.get(model_id, 0) + 1
                alias = f"{base_alias}_{survival_count[model_id]}"

                if base_alias == str(current_alias):
                    current_alias += 1

                alias_map[(model_id, gen_index)] = alias

                G, edge_color_map = self.create_node(G, model_info, alias, alias_map, displayed_models_this_gen, gen_index, model_id, edge_color_map)

                if model_id == best_model_id:
                    color_map.append('blue')
                elif survival_count[model_id] > 1:
                    color_map.append('green')
                else:
                    color_map.append('red')

            horizontal_positions = [(i - (len(displayed_models_this_gen) - 1) / 2) for i in range(len(displayed_models_this_gen))]
            for alias in displayed_models_this_gen:
                pos[alias] = (horizontal_positions.pop(0), -gen_index)
                node_sizes.append(3000 / (len(all_ancestors) / len(self.genealogy_info)))

        nx.draw(G, pos, node_color=color_map, edge_color=edge_color_map, node_size=node_sizes, with_labels=True, arrows=True)
        plt.show()