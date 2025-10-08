import random
from typing import List, Tuple
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from utils_snapshot import population_snapshot
from ga_gene_blocks import split_genes, join_genes



Individual = str


Population = List[Individual]


def initialize_population(size: int, seed: int)-> List[str]:
    random.seed(seed)
    population = [''.join(random.choices('01', k=23)) for _ in range(size)]
    initial_population = population_snapshot(population,seed)
    return initial_population


def tournament_selection(population:Population, fitness:List[int], num_selected:int, k:int):

    selected = []

    for _ in range(num_selected):
        contestant_indices = random.choices(range(len(population)), k=k)
        contestant_fitness = [fitness[i] for i in contestant_indices]
        contestants = [population[i] for i in contestant_indices]
        max_index = contestant_fitness.index(max(contestant_fitness))
        winner = contestants[max_index]
        selected.append(winner)
    return selected

def elitism(population: Population,
            fitness_values: List[float],
            num_elites: int) -> Population:

    paired = list(zip(population, fitness_values))

    sorted_pairs = sorted(paired, key=lambda x: x[1], reverse=True)

    sorted_individuals = [ind for ind, fit in sorted_pairs]

    return sorted_individuals[:num_elites]

def gene_block_crossover(parent1: Individual, parent2: Individual,cross_rate: float = 0.5) -> Tuple[Individual, Individual]:

    p1_genes = split_genes(parent1)
    p2_genes = split_genes(parent2)

    child1_genes, child2_genes = [], []

    for gene_idx in range(5):

        if random.random() < cross_rate:
            child1_genes.append(p2_genes[gene_idx])
            child2_genes.append(p1_genes[gene_idx])
        else:
            child1_genes.append(p1_genes[gene_idx])
            child2_genes.append(p2_genes[gene_idx])

    return join_genes(child1_genes), join_genes(child2_genes)

def bit_flip_mutation(individual: Individual, mutation_rate: float = 0.2) -> Individual:

    mutated = list(individual)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = '1' if mutated[i] == '0' else '0'
    return ''.join(mutated)

def genetic_algorithm(population:List[str], fitness:List[float], cross_rate: float, mutation_rate: float, seed: int) -> List[str]:

    fitness_values = fitness

    tournament_parents = tournament_selection(population, fitness_values, num_selected=9, k=4)
    elite_parents = elitism(population, fitness_values, num_elites=3)
    all_parents = [p[:] for p in tournament_parents + elite_parents]

    offspring = []
    while len(offspring) < 36:
        parent_a, parent_b = random.choices(all_parents, k=2)
        child_a, child_b = gene_block_crossover(parent_a[:], parent_b[:],cross_rate)
        child_a = bit_flip_mutation(child_a[:], mutation_rate)
        child_b = bit_flip_mutation(child_b[:], mutation_rate)

        if len(offspring) + 2 <= 36:
            offspring.extend([child_a, child_b])
        else:
            offspring.append(child_a)
    next_population = population_snapshot(offspring,seed)  # 更新种群

    return next_population


