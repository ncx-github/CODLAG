"""
Auther: ncx@stu.ouc.edu.cn
Last Revision: 2023/10/20
Requirements:
    numpy 1.24.1
    pandas 1.5.3
    python 3.9.16
    scikit-learn 1.2.2
    scipy 1.10.1
    torch 1.13.1+cu116
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import r2_score
import time
from utils import average_without_extremes

# training parameters
try_nums = 10
speed = 24
selected_features_input = ""  # Select input variables
target = '' # label
# hyperparameters
num_epochs = 50
window_size = 20
train_scale = 0.7
batch_size = 32
hidden_size = 64
num_layers = 5
learning_r = 0.001

filename = ""  # dataset in csv file
data = pd.read_csv(filename)

selected_features = [int(idx) for idx in selected_features_input.split()]
selected_feature_names = [f'X{idx}' for idx in selected_features]
X = data[selected_feature_names].values
y = data[target].values  # Label

X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

X_window = []
y_window = []
for i in range(len(data) - window_size):
    X_window.append(X[i:i + window_size])
    y_window.append(y[i + window_size])
X_window = np.array(X_window)
y_window = np.array(y_window)
X_tensor = torch.tensor(X_window, dtype=torch.float32)
y_tensor = torch.tensor(y_window, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(train_scale * len(dataset))
test_size = len(dataset) - train_size

# try gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# LSTM
class LSTMModel(nn.Module):
    def __init__(self, inputsize, hiddensize, numlayers, outputsize):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(inputsize, hiddensize, numlayers, batch_first=True)
        self.fc = nn.Linear(hiddensize, outputsize)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


r2_score_ = [0 for i in range(try_nums)]
accuracy_ = [0 for i in range(try_nums)]
elapsed_time_ = [0 for i in range(try_nums)]
for try_num in range(0, try_nums):
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_size = len(selected_feature_names)
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_r)

    print(f'---------- Try {try_num + 1} ----------')
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        i2 = 0
        for i, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if epoch % 10 == 0 and i == len(train_loader):
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / i:.4f}')
            i2 = i
        if epoch + 1 == num_epochs:
            print(f'Last Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / i2:.4f}')
    end_time = time.time()
    elapsed_time_[try_num] = end_time - start_time
    print(f'Training Time: {elapsed_time_[try_num]:.4f}s')

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    r2 = r2_score(y_true, y_pred)
    r2_score_[try_num] = r2
    print(f'R2 Score: {r2:.8f}')

if try_nums > 2:
    r2_score_mean = average_without_extremes(r2_score_)
else:
    r2_score_mean = np.mean(r2_score_)

elapsed_time = np.mean(elapsed_time_)
total_time = np.sum(elapsed_time_)
print(f'----- {try_nums} Tries in total -----')
print(f'Average R2 Score: {r2_score_mean:.8f}')
print(f'Average Training Time: {elapsed_time:.4f}s')
print(f'Total Training Time: {total_time:.4f}s')
print(f'---------- Over! ----------')
