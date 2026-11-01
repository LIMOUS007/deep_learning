import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8), 
            nn.LeakyReLU(0.01),
            nn.Linear(8, 8),
            nn.LeakyReLU(0.01),
            nn.Linear(8, 3)
        )
    
    def forward(self, x):
        return self.network(x)
    
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
def validate(model, loader, loss_function):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            loss = loss_function(logits, y_batch)
            total_loss += loss.item() * len(x_batch)
    
    return total_loss / len(loader.dataset)

def confusion_matrix_multiclass(predictions, targets, num_classes):
    matrix = torch.zeros(num_classes, num_classes, dtype = torch.int64)
    for actual, predicted in zip(targets, predictions):
        matrix[actual, predicted] += 1
    return matrix

def evaluate_multiclass(model, loader, num_classes):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            predictions = torch.argmax(logits, dim = 1)
            all_predictions.append(predictions)
            all_targets.append(y_batch)
        
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)
    accuracy = (predictions == targets).float().mean()
    cm = confusion_matrix_multiclass(predictions, targets, num_classes)
    precision = []
    recall = []
    f1 = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        class_precision = tp / (tp + fp + 1e-8)
        class_recall = tp / (tp + fn + 1e-8)
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall + 1e-8)
        precision.append(class_precision)
        recall.append(class_recall)
        f1.append(class_f1)
    precision = torch.stack(precision)
    recall = torch.stack(recall)
    f1 = torch.stack(f1)
    return accuracy, precision, recall, f1, cm

x = torch.tensor([
    [1., 1.],
    [1., 2.],
    [2., 1.],
    [2., 2.],

    [5., 5.],
    [5., 6.],
    [6., 5.],
    [6., 6.],

    [9., 9.],
    [9., 10.],
    [10., 9.],
    [10., 10.]
])

y = torch.tensor([
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2
])

x_val = torch.tensor([
    [1.5, 1.5],
    [5.5, 5.5],
    [9.5, 9.5]
])

y_val = torch.tensor([
    0,
    1,
    2
])

dataset = MyDataset(x, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
val_dataset = MyDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
model = NeuralNetwork()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(2000):
    model.train()
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_function(logits, y_batch)
        loss.backward()
        optimizer.step()
        
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            logits = model(x)
            predictions = torch.argmax(logits, dim=1)
        
        print(f"Epoch {epoch}: " f"Loss = {loss.item():.4f}")
        # print(f"Logits:", logits)
        # print(f"Predictions:", predictions)
        # print("Actual:", y)
        accuracy = (predictions == y).float().mean()
        print("Accuracy:", accuracy.item())
        
accuracy, precision, recall, f1, cm = evaluate_multiclass(model, val_loader, num_classes=3)
print(f"Accuracy: {accuracy:.4f}")

for c in range(3):
    print(f"\nClass {c}")
    print(f"Precision: {precision[c]:.4f}")
    print(f"Recall:    {recall[c]:.4f}")
    print(f"F1:        {f1[c]:.4f}")

print("\nConfusion Matrix:")
print(cm)
# -------------------------
# MACRO
# -------------------------

macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = f1.mean()

print("\nMacro Average")
print(f"Precision: {macro_precision:.4f}")
print(f"Recall:    {macro_recall:.4f}")
print(f"F1 Score:  {macro_f1:.4f}")


# -------------------------
# WEIGHTED
# -------------------------

class_counts = cm.sum(dim=1)

weighted_precision = (
    precision * class_counts
).sum() / class_counts.sum()

weighted_recall = (
    recall * class_counts
).sum() / class_counts.sum()

weighted_f1 = (
    f1 * class_counts
).sum() / class_counts.sum()

print("\nWeighted Average")
print(f"Precision: {weighted_precision:.4f}")
print(f"Recall:    {weighted_recall:.4f}")
print(f"F1 Score:  {weighted_f1:.4f}")


# -------------------------
# MICRO
# -------------------------

tp_total = torch.diag(cm).sum()

fp_total = cm.sum(dim=0) - torch.diag(cm)
fn_total = cm.sum(dim=1) - torch.diag(cm)

micro_precision = (
    tp_total /
    (tp_total + fp_total.sum() + 1e-8)
)

micro_recall = (
    tp_total /
    (tp_total + fn_total.sum() + 1e-8)
)

micro_f1 = (
    2 * micro_precision * micro_recall /
    (micro_precision + micro_recall + 1e-8)
)

print("\nMicro Average")
print(f"Precision: {micro_precision:.4f}")
print(f"Recall:    {micro_recall:.4f}")
print(f"F1 Score:  {micro_f1:.4f}")