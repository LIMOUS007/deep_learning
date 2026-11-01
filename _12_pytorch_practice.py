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
        prediction = model(x_batch)
        loss = loss_function(prediction, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x_batch)
    
    return total_loss / len(loader.dataset)
def validate(model, loader, loss_function):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            prediction = model(x_batch)
            loss = loss_function(prediction, y_batch)
            total_loss += loss.item() * len(x_batch)
            
    return total_loss / len(loader.dataset)

x = torch.tensor([
    [1., 2.],
    [2., 3.],
    [3., 4.],
    [4., 5.],
    [5., 6.],
    [6., 7.],
    [7., 8.],
    [8., 9.],
])

y = torch.tensor([
    [4.],
    [7.],
    [10.],
    [13.],
    [16.],
    [19.],
    [22.],
    [25.]
])

x_val = torch.tensor([
    [9., 10.],
])

y_val = torch.tensor([
    [28.],
])
 
train_history = []
val_history = []               
dataset = MyDataset(x, y)
loader = DataLoader(dataset, batch_size = 4, shuffle = True)
model = NeuralNetwork()
loss_function = nn.MSELoss()
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
        print(f"New best validation loss: {best_val_loss:.6f}")
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