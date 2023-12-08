import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from dataset import dataset

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size=128, output_size=2, num_layers=2, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        # Set batch_first=True so that the input is expected to be of shape [batch_size, seq_length, features]
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        # Reshape x to [batch_size, 1, input_size] if it is not already
        if x.dim() < 3:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
<<<<<<< HEAD
        lstm_out = self.dropout(lstm_out[:, -1, :]) # Use only the last output of the sequence
=======
        lstm_out = self.dropout(lstm_out[:, -1, :])
>>>>>>> 19d386be5a62afddb89e8ee136e046014f647494
        out = self.linear(lstm_out)
        return out

# Initialize Dataset
ds = dataset()

# Prepare Data
def prepare_data(ds, dataset_type):
    df = ds(dataset="LIAR", type=dataset_type)
    X = df[[i for i in df.columns if "vector_" in i]].values
    y = df["label"].values
    return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64))

train_dataset = prepare_data(ds, "train")
val_dataset = prepare_data(ds, "val")
test_dataset = prepare_data(ds, "test")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_X, val_y = val_dataset.tensors
test_X, test_y = test_dataset.tensors

# Get input size from train_dataset
train_X_example, _ = train_dataset[0]
input_size = train_X_example.shape[0]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Loss, and Optimizer
model = LSTMModel(input_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the Model
def train_model(model, train_loader, val_X, val_y, epochs=10):
    model.train()
    train_loss_records = []
    val_loss_records = []

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                 torch.zeros(1, 1, model.hidden_layer_size).to(device))

            output = model(inputs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

        train_loss_records.append(loss.item())

        with torch.no_grad():
            val_pred = model(val_X.to(device))
            val_loss = criterion(val_pred, val_y.to(device)).item()
            val_accuracy = accuracy_score(val_y.numpy(), np.argmax(val_pred.cpu().numpy(), axis=1))

            print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            val_loss_records.append(val_loss)

    return train_loss_records, val_loss_records

train_loss, val_loss = train_model(model, train_loader, val_X, val_y, epochs=10)

# Test the Model
with torch.no_grad():
    test_pred = model(test_X.to(device))
    test_pred_labels = np.argmax(test_pred.cpu().numpy(), axis=1)
    test_accuracy = accuracy_score(test_y.numpy(), test_pred_labels)
    test_f1 = f1_score(test_y.numpy(), test_pred_labels)

    print(f"Test Accuracy: {test_accuracy}, Test F1 Score: {test_f1}")

    cm = confusion_matrix(test_y.numpy(), test_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("./plots/lstm_cm.png")

# Plotting Loss
plt.figure()
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./plots/lstm_loss.png')
plt.show()
