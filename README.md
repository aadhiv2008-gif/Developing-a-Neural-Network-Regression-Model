# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Aadhithya.V

### Register Number:212225040002

```
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("/content/Exp-1.csv")
df


x = df[["Input"]].values
y = df[["Output"]].values
xt,xst,yt,yst = train_test_split(x,y,test_size=0.2,random_state=42)
scale1 = MinMaxScaler()
scale2=MinMaxScaler()
xt = scale1.fit_transform(xt)
xst = scale2.fit_transform(xst)
xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)
yt = torch.FloatTensor(yt)
yst = torch.FloatTensor(yst)

class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
    def forward(self,x):
        return self.network(x)
model = neuralnet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
epochs = 1000
losses=[]
for i in range(epochs):
    optimizer.zero_grad()
    pred = model(xt)
    loss = criterion(pred, yt)
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"{i}/{epochs} Loss: {loss.item():.4f}")
        losses.append(loss.item())
new = scale1.transform([[16]])
new = torch.FloatTensor(new)

pred = model(new)
print(pred.item())

with torch.no_grad():
    pred=model(xst)
    loss_test=criterion(pred,yst)
    print(loss_test)

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

```


### Dataset Information
<img width="182" height="432" alt="685899054-14fbc953-f1c4-4350-9e8f-637ee9bda3ac" src="https://github.com/user-attachments/assets/3c4fef0e-c52f-44e5-a5c9-6891174b759c" />

### OUTPUT

<img width="540" height="502" alt="685899405-00ffb9c2-d72a-45b2-b5dc-f80749aae8c0" src="https://github.com/user-attachments/assets/918890bb-0b4a-410c-9038-f64b6e270d34" />

### Training Loss Vs Iteration Plot

<img width="645" height="520" alt="685900606-88bcae62-7941-4398-af21-f7ac4756f514" src="https://github.com/user-attachments/assets/2019abc3-0cb1-4ae9-aa1d-9f1107d6cbab" />

### New Sample Data Prediction

<img width="605" height="97" alt="685900445-2e67eb10-2f03-4d81-b529-3852013e622b" src="https://github.com/user-attachments/assets/902440d4-2200-4647-9774-6e9202f3e19e" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
