from genetic_step import genetic_algorithm
from utils_csv_to_list import read_csv_to_list
from element_binary_to_proportion import binary_to_proportion
from mlp_step import get_model_predictions


def closed_loop_search(model_path, current_population, num_iteration):

    binary_population = read_csv_to_list("population.csv")
    population_fitness = read_csv_to_list("fitness.csv")

    max_iteration = current_population + num_iteration

    while current_population < max_iteration:
        current_population += 1
        binary_population = genetic_algorithm(
                binary_population,
                population_fitness,
                0.5,
                0.2,
                current_population
            )
        proportion_population = binary_to_proportion(binary_population)
        population_fitness =get_model_predictions(proportion_population, model_path)

        print(binary_population)
        print(proportion_population)
        print(population_fitness)


