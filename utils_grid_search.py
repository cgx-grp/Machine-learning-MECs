import numpy as np
import itertools
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from utils_kfold_validation import kfold_validation

class GridSearch:
    def __init__(self, input_dim, output_dim, input_features, output_labels, k_folds=10):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_features = input_features
        self.output_labels = output_labels
        self.kf = KFold(n_splits=k_folds, shuffle=True, random_state=0)

    def run(self, hidden_dim_options, num_hidden_layer_options, learning_rates, num_epochs=200):
        all_combinations = list(itertools.chain.from_iterable(
            itertools.product(hidden_dim_options, repeat=n) for n in num_hidden_layer_options
        ))

        for hidden_dims in all_combinations:
            for lr in learning_rates:
                kfold_validation(input_features=self.input_features,
                                 output_labels=self.output_labels,
                                 hidden_dims=hidden_dims,
                                 lr=lr,
                                 num_epochs=num_epochs,
                                 use_early_stop=0)
