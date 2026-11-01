import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ignore Tcl/Tk paths left over from a different Python installation.
os.environ.pop("TCL_LIBRARY", None)
os.environ.pop("TK_LIBRARY", None)

from matplotlib import pyplot as plt

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # self.layer1 = nn.Linear(2, 4)
        # self.layer2 = nn.Linear(4, 4)
        # self.layer3 = nn.Linear(4, 1)
        # self.activation = nn.LeakyReLU(0.01)
        self.network = nn.Sequential(
            nn.Linear(2, 4),
            nn.LeakyReLU(0.01),
            nn.Linear(4, 4),
            nn.LeakyReLU(0.01),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        return self.network(x)
    #     x = self.activation(self.layer1(x))
    #     x = self.activation(self.layer2(x))
    #     x = self.layer3(x)
    #     return x
    
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    


def train_one_epoch(model, loader, loss_function, optimizer):
    model.train()
    total_loss = 0
    
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_function(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x_batch)
    
    return total_loss / len(loader.dataset)

def validate(model, loader, loss_function):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            loss = loss_function(logits, y_batch)
            total_loss += loss.item() * len(x_batch)
            
    return total_loss / len(loader.dataset)

def evaluate_classification(model, loader):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()
            all_predictions.append(predictions)
            all_targets.append(y_batch)
        
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    accuracy = (predictions == targets).float().mean()
    tp = ((predictions == 1) & (targets == 1)).sum()
    tn = ((predictions == 0) & (targets == 0)).sum()
    fp = ((predictions == 1) & (targets == 0)).sum()
    fn = ((predictions == 0) & (targets == 1)).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1, tp, tn, fp, fn

def evaluate_threshold(model, loader, threshold):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).float()
            all_predictions.append(predictions)
            all_targets.append(y_batch)
        
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    accuracy = (predictions == targets).float().mean()
    tp = ((predictions == 1) & (targets == 1)).sum()
    tn = ((predictions == 0) & (targets == 0)).sum()
    fp = ((predictions == 1) & (targets == 0)).sum()
    fn = ((predictions == 0) & (targets == 1)).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1, tp, tn, fp, fn

x = torch.tensor([
    [1., 1.],
    [2., 1.],
    [1., 2.],
    [2., 2.],
    [8., 8.],
    [9., 8.],
    [8., 9.],
    [9., 9.]
])

y = torch.tensor([
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [1.],
    [1.]
])

x_val = torch.tensor([
    [3., 3.],
    [10., 10.]
])

y_val = torch.tensor([
    [0.],
    [1.]
])
 
train_history = []
val_history = []               
dataset = MyDataset(x, y)
loader = DataLoader(dataset, batch_size = 4, shuffle = True)
model = NeuralNetwork()
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
val_dataset = MyDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
best_val_loss = float("inf")
patience = 100
patience_counter = 0


for epoch in range(2000):
    train_loss = train_one_epoch(model, loader, loss_function, optimizer)
    val_loss = validate(model, val_loader, loss_function)
    train_history.append(train_loss)
    val_history.append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model.pth")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    if epoch % 100 == 0:
        print(
        f"Epoch {epoch}: "
        f"Training Loss: {train_loss:.4f}, "
        f"Validation Loss: {val_loss:.4f}"
    )

plt.plot(train_history, label="Train")
plt.plot(val_history, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
model.load_state_dict(torch.load("model.pth"))  
accuracy, precision, recall, f1, tp, tn, fp, fn = evaluate_classification(model, val_loader)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"True Positives: {tp}")
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    accuracy, precision, recall, f1, tp, tn, fp, fn  = evaluate_threshold(model, val_loader, threshold)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
