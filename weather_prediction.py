import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv("weather.csv")
df = df.dropna() #Removes any row with missing values

X = df.drop(columns=["RainTomorrow", "RISK_MM"]) #Separate features from target. "RISK_MM" is the amount of rain the next day so including it would cause a data leakage
y = df["RainTomorrow"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#convert to tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #Data loader serves them in batches of 32
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(in_features=17, out_features=10),
    nn.ReLU(), #Replaces negatives with 0
    nn.Linear(in_features=10, out_features=5),
    nn.ReLU(),
    nn.Linear(in_features=5, out_features=1),
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #adjusts weights to reduce loss

def accuracy_fn(y_true, y_pred):
    """Counts the number of correct predictions and returns the accuracy."""
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

epochs = 100

#training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for batch, (X_batch, y_batch) in enumerate(train_loader):
        y_logits = model(X_batch).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits)) #sigmoid converts logits to probabilities (0 to 1)

        loss = loss_fn(y_logits, y_batch)
        train_loss += loss.item()
        acc = accuracy_fn(y_true=y_batch, y_pred=y_pred)
        train_acc += acc

        optimizer.zero_grad() #clears gradients left over from the previous batch
        loss.backward() #backpropogation. Computes how much each weight contributed to the error
        optimizer.step() #nudges each weight in the direction that reduces the loss



    avg_train_loss = train_loss / (batch + 1)
    avg_train_acc = train_acc / (batch + 1)


    model.eval()
    test_loss = 0.0
    total_test_acc = 0.0
    for batch, (X_batch, y_batch) in enumerate(test_loader):
      with torch.inference_mode():
        test_logits = model(X_batch).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        loss = loss_fn(test_logits, y_batch)
        test_loss += loss.item()
        test_acc = accuracy_fn(y_true=y_batch, y_pred=test_pred)
        total_test_acc += test_acc
    avg_test_loss = test_loss / (batch + 1)
    avg_test_acc = total_test_acc / (batch + 1)

    if epoch % 10 == 0:
      print(f"Epoch: {epoch} | Loss: {avg_train_loss:.5f}, Acc: {avg_train_acc:.2f}% | Test Loss: {avg_test_loss:.5f}, Test Acc: {avg_test_acc:.2f}%")

