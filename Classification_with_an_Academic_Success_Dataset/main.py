import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import torch 
import torch.nn as nn
from torch import Generator
import os

# Ignore Tcl/Tk paths left over from a different Python installation.
os.environ.pop("TCL_LIBRARY", None)
os.environ.pop("TK_LIBRARY", None)

import matplotlib.pyplot as plt


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class NeuralNetwork(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

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

def evaluate_multiclass(model, loader, num_classes):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            predictions = torch.argmax(logits, dim=1)
            all_preds.append(predictions)
            all_targets.append(y_batch)
    
    predictions = torch.cat(all_preds)
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
        class_f1 = (2 * class_precision * class_recall) / (class_precision + class_recall + 1e-8)
        precision.append(class_precision)
        recall.append(class_recall)
        f1.append(class_f1)
        
    precision = torch.stack(precision)
    recall = torch.stack(recall)
    f1 = torch.stack(f1)
    return accuracy, precision, recall, f1, cm

def confusion_matrix_multiclass(predictions, targets, num_classes):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for actual, predicted in zip(targets, predictions):
        matrix[actual, predicted] += 1
    
    return matrix


df = pd.read_csv(r"C:\coding\MACHINE\deep_learning\Classification_with_an_Academic_Success_Dataset\train.csv")
# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.dtypes)
# print(df.isna().sum())
# print(df["Target"].value_counts())
# print(df.nunique())
# print(df.head())
# print(df.tail())
# print(df["Target"].unique())

X = df.drop("Target", axis=1)
y = df["Target"]
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
dataset = MyDataset(X, y)

train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))
test_size = len(df) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator = Generator().manual_seed(42))
train_indices = train_dataset.indices
val_indices = val_dataset.indices
test_indices = test_dataset.indices
X_train = X[train_indices]
X_val = X[val_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_val = y[val_indices]
y_test = y[test_indices]
scaler = StandardScaler()
x_train = scaler.fit_transform(X_train.numpy())
x_val = scaler.transform(X_val.numpy())
x_test = scaler.transform(X_test.numpy())
X_train = torch.tensor(x_train, dtype=torch.float32)
X_val = torch.tensor(x_val, dtype=torch.float32)
X_test = torch.tensor(x_test, dtype=torch.float32)
train_dataset = MyDataset(X_train, y_train)
val_dataset = MyDataset(X_val, y_val)
test_dataset = MyDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_features = X_train.shape[1]
num_classes = len(torch.unique(y))
model = NeuralNetwork(input_features, num_classes)
class_counts = torch.bincount(y_train)
class_weights = len(y_train) / (num_classes * class_counts.float())
print("Class counts:", class_counts)
print("Class weights:", class_weights)
loss_function = nn.CrossEntropyLoss(weight = class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 5)


train_history = []
val_history = []
best_val_loss = float('inf')
patience = 20 
patience_counter = 0
for epoch in range(200):
    train_loss = train_one_epoch(model, train_loader, loss_function, optimizer)
    val_loss = validate(model, val_loader, loss_function)
    scheduler.step(val_loss)
    train_history.append(train_loss)
    val_history.append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

model.load_state_dict(torch.load("best_model.pth"))
plt.plot(train_history, label="Train Loss")
plt.plot(val_history, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()
accuracy, precision, recall, f1, cm = evaluate_multiclass(model, test_loader, num_classes)
print(f"Test Accuracy: {accuracy:.4f}")
for c in range(num_classes):
    print(f"Class {c}: Precision = {precision[c]:.4f}, Recall = {recall[c]:.4f}, F1 Score = {f1[c]:.4f}")
print("Confusion Matrix:")
print(cm)

macro_precision = precision.mean()
macro_recall = recall.mean()
macro_f1 = f1.mean()
print("\nMacro Average")
print(f"Precision: {macro_precision:.4f}")
print(f"Recall:    {macro_recall:.4f}")
print(f"F1 Score:  {macro_f1:.4f}")

class_counts = cm.sum(dim=1)
weighted_precision = (precision * class_counts).sum() / class_counts.sum()
weighted_recall = (recall * class_counts).sum() / class_counts.sum()
weighted_f1 = (f1 * class_counts).sum() / class_counts.sum()
print("\nWeighted Average")
print(f"Precision: {weighted_precision:.4f}")
print(f"Recall:    {weighted_recall:.4f}")
print(f"F1 Score:  {weighted_f1:.4f}")

tp_total = torch.diag(cm).sum()
fp_total = cm.sum(dim=0) - torch.diag(cm)
fn_total = cm.sum(dim=1) - torch.diag(cm)
micro_precision = tp_total / (tp_total + fp_total.sum() + 1e-8)
micro_recall = tp_total / (tp_total + fn_total.sum() + 1e-8)
micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-8)
print("\nMicro Average")
print(f"Precision: {micro_precision:.4f}")
print(f"Recall:    {micro_recall:.4f}")
print(f"F1 Score:  {micro_f1:.4f}")

print("\nClass Mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"Class {i}: {label}")

test_loss = validate(model, test_loader, loss_function)
print(f"Test Loss: {test_loss:.4f}")

model.eval()
with torch.no_grad():
    logits = model(X_test)
    predictions = torch.argmax(logits, dim=1)

actual_labels = encoder.inverse_transform(y_test.numpy())
predicted_labels = encoder.inverse_transform(predictions.numpy())
print("\nSample Predictions:")
for actual, predicted in zip(actual_labels[:20], predicted_labels[:20]):
    print(f"Actual: {actual}, Predicted: {predicted}")
    
with torch.no_grad():
    logits = model(X_test)
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

print("\nSample Predictions:")
for i in range(10):
    print(f"Actual: {actual_labels[i]} | " f"Predicted: {predicted_labels[i]} | " f"Confidence: {probabilities[i,  predictions[i]].item():.4f}")
    
