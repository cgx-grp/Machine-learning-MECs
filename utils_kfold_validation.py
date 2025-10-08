from torch import nn, optim
import numpy as np
from sklearn.model_selection import KFold
from initiallize_nn_model import NeuralNet
from mlp_step import train_one_epoch, evaluate_model, save_checkpoint
from data_loaders import get_data_loaders

def kfold_validation(input_features, output_labels,
                     input_dim=5, hidden_dims=3, output_dim=1,
                     lr=0.01, num_epochs=4, patience=20,
                     n_splits=10, use_early_stop=1):

    kfold = KFold(n_splits, shuffle=True, random_state=0)
    fold_mse_results = []
    plt.figure(figsize=(10, 8))
    for fold_num, (train_idx, val_idx) in enumerate(kfold.split(input_features), 1):
        print(f"\nStarting Fold {fold_num}")
        model = NeuralNet(input_dim, hidden_dims, output_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_loader, val_loader = get_data_loaders(input_features, output_labels, train_idx, val_idx)

        best_val_loss = float('inf')
        early_stop_counter = 0

        for epoch in range(num_epochs):
            train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss = evaluate_model(model, val_loader, criterion)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                checkpoint = {
                    'input_dim': input_dim,
                    'hidden_dims': hidden_dims,
                    'output_dim': output_dim,
                    'learning_rate': lr,
                    'best_val_loss': val_loss,
                }
                save_checkpoint(f".checkpoints/fold{fold_num}_loss{best_val_loss:.6f}.pth", model, checkpoint)
            elif use_early_stop == 1:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            print(f"Fold {fold_num}, Epoch {epoch + 1}, Loss: {val_loss:.6f}")
        fold_mse_results.append(best_val_loss)
        print(
            f"[Fold {fold_num}] Hidden: {hidden_dims}, LR: {lr}, Best Val Loss: {best_val_loss:.6f}")
    avg_mse = np.mean(fold_mse_results)
    print(f"==> Avg MSE for {hidden_dims} @ LR={lr}: {avg_mse:.4f}\n")

def train_kfold(input_features,
                output_labels,
                input_dim, hidden_dims, output_dim,
                lr, num_epochs, patience=20,
                n_splits=10, use_early_stop=1):

    kfold_validation(input_features, output_labels,
                     input_dim, hidden_dims, output_dim,
                     lr, num_epochs, patience,
                     n_splits, use_early_stop)