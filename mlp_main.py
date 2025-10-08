import torch
from torch import nn
from torch import optim
import numpy as np
from element_binary_to_proportion import binary_to_proportion
from ga_integrated_mlp import closed_loop_search
from utils_csv_to_list import read_csv_to_list
from utils_csv_to_list import merged_csv_to_list
from utils_grid_search import GridSearch
from utils_kfold_validation import train_kfold

input_populations = binary_to_proportion(read_csv_to_list("data/compilation_four_populations.csv"))
file_paths = (
    'data/first_population_fitness.csv',
    'data/second_population_fitness.csv',
    'data/third_population_fitness.csv',
    'data/fourth_population_fitness.csv')
output_fitness = np.array(merged_csv_to_list(file_paths), dtype=np.float32)

input_features = torch.tensor(input_populations, dtype=torch.float32)
output_labels = torch.tensor(output_fitness, dtype=torch.float32).view(-1, 1)

search = GridSearch(5,
                    1,
                    input_features,
                    output_labels)
search.run(
    hidden_dim_options=[10, 15, 20, 25, 30, 35, 40, 45, 50],
    num_hidden_layer_options=[2, 3, 4],
    learning_rates=[0.005, 0.01, 0.02],
    num_epochs=200
)

train_kfold(input_features,
            output_labels,
            5,
            (25,20,35),
            1,
            0.01,
            400)

closed_loop_search("data/mlp_model.pth", 4, 100)
