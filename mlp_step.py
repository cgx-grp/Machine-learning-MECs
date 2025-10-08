from torch import nn
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from typing import List, Tuple
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from reload_model import get_pre_trained_model

def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    predictions, true_values = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predictions.extend(outputs.view(-1).tolist())
            true_values.extend(labels.view(-1).tolist())
    val_loss /= len(val_loader)
    return val_loss


def save_checkpoint(path, model, meta_info):
    torch.save({
        'model_state_dict': model.state_dict(),
        **meta_info
    }, path)


def get_model_predictions(element_proportions_np: np.ndarray, model_path: str, population_num) -> List[float]:

    # Convert the input NumPy array to a PyTorch Tensor
    element_proportion_tensor = torch.tensor(element_proportions_np, dtype=torch.float32)

    # Create a DataLoader for inference. Labels are dummy as they are not used for prediction.
    # Assuming batch_size=12 as in your original code.
    element_loader = DataLoader(TensorDataset(element_proportion_tensor, torch.zeros_like(element_proportion_tensor)), batch_size=5, shuffle=False)

    performance_predictions = []

    # Disable gradient calculation for inference to save memory and speed up computation
    with torch.no_grad():
        for features, _ in element_loader: # _ for dummy labels
            outputs = get_pre_trained_model(model_path, features, population_num)
            performance_predictions.extend(outputs)

    return performance_predictions