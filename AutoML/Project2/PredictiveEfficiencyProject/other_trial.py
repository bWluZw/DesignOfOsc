import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# Auxiliary functions to read dataset

def excel2dict(filename):
    xls = pd.ExcelFile(filename)
    df = xls.parse(xls.sheet_names[0])
    data = df.to_dict()

    for key, value in data.items():
        data[key] = np.array(list(value.values()))

    return data


def excel2numpy(filename):
    xls = pd.ExcelFile(filename)
    df = xls.parse(xls.sheet_names[0])
    data = df.to_dict()

    sortednames = sorted(data.keys(), key=lambda x: x.lower())

    data_x = []
    data_y = []

    for key in sortednames:
        if key == 'PCE':
            data_y.append(np.array(list(data[key].values())))
        else:
            data_x.append(np.array(list(data[key].values())))

    return np.array(data_x).T, np.array(data_y).T


# Custom correlation coefficient metric

def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x)
    my = torch.mean(y)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
    r = r_num / r_den
    r = torch.clamp(r, min=-1.0, max=1.0)
    return r ** 2


# Neural network model

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Method to load data

def load_data(path):
    x_train, y_train = excel2numpy(path)
    x_test, y_test = excel2numpy(path)

    # Standardize the data (optional but recommended for better performance)
    scaler_x = StandardScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)

    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for batching
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    test_data = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_loader, test_loader


# Method to train and evaluate the model

def train_and_evaluate(model, train_loader, test_loader, num_epochs=100):
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters())

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_r2 = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Calculate the correlation coefficient
            train_r2 += correlation_coefficient(labels, outputs).item()

        train_loss = running_loss / len(train_loader)
        train_r2 /= len(train_loader)

        # Evaluate the model
        model.eval()
        test_loss = 0.0
        test_r2 = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                test_r2 += correlation_coefficient(labels, outputs).item()

        test_loss /= len(test_loader)
        test_r2 /= len(test_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train R2: {train_r2:.4f} - "
              f"Test Loss: {test_loss:.4f}, Test R2: {test_r2:.4f}")

    return model


# Main entry point

def main():
    # Load data
    train_loader, test_loader = load_data()

    # Build model
    input_size = train_loader.dataset[0][0].shape[0]  # Input size is the number of features
    model = NeuralNetwork(input_size)

    # Train and evaluate the model
    trained_model = train_and_evaluate(model, train_loader, test_loader)

    # Save the model
    # torch.save(trained_model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
