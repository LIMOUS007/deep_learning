import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
 
                    
dataset = MyDataset(x, y)
loader = DataLoader(dataset, batch_size = 4, shuffle = True)
model = NeuralNetwork()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for epoch in range(2000):
    model.train()
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        prediction = model(x_batch)
        loss = loss_function(prediction, y_batch)
        loss.backward()
        optimizer.step()
        
    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            val_prediction = model(x_val)
            val_loss = loss_function(val_prediction, y_val)
        print(f'Epoch {epoch}: Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
        print(f'Validation Prediction: {val_prediction.item():.4f}, Actual Value: {y_val.item():.4f}')

torch.save(model.state_dict(), "model.pth")
for name, parameter in model.named_parameters():
    print(name, parameter.shape)