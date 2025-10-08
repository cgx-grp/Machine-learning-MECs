import torch
from torch._C import device

from initiallize_nn_model import NeuralNet
from utils_snapshot import fitness_snapshot


def get_pre_trained_model(model_path, population,population_num):
    # Load the model checkpoint
    checkpoint = torch.load(model_path)

    # Re-instantiate the model with its saved architecture
    input_dim = checkpoint['input_dim']
    hidden_dims = checkpoint['hidden_dims']
    output_dim = checkpoint['output_dim']
    loaded_model = NeuralNet(input_dim, hidden_dims, output_dim)

    # Load the model weights
    loaded_model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode (important for disabling dropout, batch norm updates, etc.)
    loaded_model.eval()
    output = loaded_model(population).detach().numpy().tolist()
    predicted_fitness = fitness_snapshot(output,population_num)
    return predicted_fitness

