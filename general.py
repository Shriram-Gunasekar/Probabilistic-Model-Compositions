import torch
import torch.nn as nn
from pomegranate import *

torch.manual_seed(42)  # Setting a seed for reproducibility

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = SimpleNN()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Creating synthetic data
X_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Training the model
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

# Using the trained model to predict
X_test = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
predictions = model(X_test)
print(predictions)

# Creating a Gaussian Mixture Model using Pomegranate
gmm = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=2, X=X_train.numpy())

# Calculating probabilities using the trained GMM
probabilities = gmm.predict_proba(X_test.numpy())
print(probabilities)
