from neural_networks.model import NeuralNetwork
from .selection import select

def initialize_population(size, architecture_func, optimizer_func, learning_rate, loss_function, metrics_function, device):
    pop = [NeuralNetwork(model=architecture_func(), optimizer_func=optimizer_func, learning_rate=learning_rate, loss_function=loss_function, metrics_function=metrics_function, device=device) for _ in range(size)]
    genealogy_info = [{mod.id: {'created_at_gen': 0, 'parents': []} for mod in pop}]
    return pop, genealogy_info

def calculate_performance_gap(population, population_size):
    performances = [model.fitness for model in sorted(population, key=lambda model: model.fitness)[:population_size // 2]]
    return max(performances) - min(performances)

def adjust_population_size(current_population_size, performance_gap, threshold):
    return int(current_population_size * (1.1 if performance_gap > threshold else 0.9 if performance_gap < threshold else 1))

def reproduce(population, crossover_func, population_size, selection_method, generation, genealogy_info, dynamic_population):
    performance_gap = calculate_performance_gap(population, population_size)
    threshold = 0.1
    if dynamic_population:
        temp_population_size = adjust_population_size(len(population), performance_gap, threshold)
        population_size = min(max(temp_population_size, population_size // 2), population_size * 5)
    nb_to_select = (1 + (1 + 8 * population_size)**0.5) // 2
    selected = select(population, selection_method)[:int(nb_to_select)]
    children = []
    gen = {parent.id: {'created_at_gen': generation-1, 'parents': [parent.id]} for parent in selected}
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if len(children) < population_size - len(selected):
                parent_ids = [selected[i].id, selected[j].id]
                child = crossover_func(selected[i], selected[j], generation, parent_ids)
                children.append(child)
                gen[child.id] = {'created_at_gen': generation, 'parents': parent_ids}
    
    genealogy_info.append(gen)
    return selected + children, genealogy_info
