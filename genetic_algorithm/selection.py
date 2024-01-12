import random

def select(population, method):
    return {
        'roulette': roulette_selection,
        'sus': stochastic_universal_selection,
        'elitism': elitism_selection
    }.get(method, lambda _: ValueError("Unknown selection method"))(population)[:len(population)//2]

def roulette_selection(population):
    random.shuffle(population)
    return population

def stochastic_universal_selection(population):
    total_fitness = sum(model.fitness for model in population)
    pointers = sorted(random.uniform(0, total_fitness) for _ in range(len(population)))
    return select_from_pointers(population, pointers)

def elitism_selection(population):
    return sorted(population, key=lambda model: model.fitness)

def select_from_pointers(population, pointers):
    selected, running_sum = set(), 0
    for pointer in pointers:
        for model in population:
            running_sum += model.fitness
            if running_sum > pointer and model not in selected:
                selected.add(model)
                break
    return list(selected)
