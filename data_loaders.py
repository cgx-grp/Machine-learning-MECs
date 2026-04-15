from torch.utils.data import TensorDataset, DataLoader

def get_data_loaders(input_features, output_labels, train_idx, val_idx, batch_size=4):

    train_data = TensorDataset(input_features[train_idx], output_labels[train_idx])
    val_data = TensorDataset(input_features[val_idx], output_labels[val_idx])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader
