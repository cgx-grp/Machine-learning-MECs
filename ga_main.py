
from genetic_step import (
    initialize_population,
    genetic_algorithm,
)
from utils_csv_to_list import read_csv_to_list


def main():
    first_population = initialize_population(36, 1)
    first_population_fitness = read_csv_to_list("data/first_population_fitness.csv")
    second_population = genetic_algorithm(
        first_population,
        first_population_fitness,
        0.5,
        0.1,
        2
    )
    print(f"第二代：{second_population}")
    second_population_fitness = read_csv_to_list("data/second_population_fitness.csv")
    third_population = genetic_algorithm(
        second_population,
        second_population_fitness,
        0.5,
        0.3,
        3
    )
    print(f"第三代：{third_population}")
    third_population_fitness = read_csv_to_list("data/third_population_fitness.csv")
    fourth_population = genetic_algorithm(third_population,
        third_population_fitness,
        0.5,
        0.2,
        4)
    fourth_population_fitness = read_csv_to_list("data/fourth_population_fitness.csv")
    print(f"第四代：{fourth_population}")


if __name__ == "__main__":
    main()