import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from lora import LoRA, LoRALinear

# Data loading and transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Loading MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', download=True, transform=transform)

# Splitting data
train_size = int(0.6 * len(mnist_dataset))
finetune_size = int(0.3 * len(mnist_dataset))
test_size = len(mnist_dataset) - train_size - finetune_size

train_data, finetune_data, test_data = random_split(mnist_dataset, [train_size, finetune_size, test_size])

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)  # Input layer
        self.fc2 = nn.Linear(100, 10)     # Output layer

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Initialize network, optimizer, and criterion
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Train the model
train(model, device, train_loader, optimizer, criterion)

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = torch.zeros(10, 10)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy, confusion_matrix

test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
initial_loss, initial_accuracy, initial_confusion = evaluate(model, device, test_loader)

# Calculate per-category accuracy
category_accuracy = initial_confusion.diag() / initial_confusion.sum(1)
worst_category = torch.argmin(category_accuracy).item()

# Filter the finetune dataset for the worst performing category
finetune_indices = [i for i, (img, label) in enumerate(finetune_data.dataset) if label == worst_category and i in finetune_data.indices]
finetune_subset = Subset(finetune_data.dataset, finetune_indices)
finetune_loader = DataLoader(finetune_subset, batch_size=64, shuffle=True)

# Replace and freeze layers
model.fc2 = LoRALinear(model.fc2, rank=10, alpha=1).to(device)
for param in model.fc1.parameters():
    param.requires_grad = False

# Finetune on the filtered dataset
train(model, device, finetune_loader, optimizer, criterion)

final_loss, final_accuracy, final_confusion = evaluate(model, device, test_loader)

# Calculate per-category accuracy after finetuning
final_category_accuracy = final_confusion.diag() / final_confusion.sum(1)

categories = list(range(10))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
axes[0].bar(categories, category_accuracy.numpy(), color='blue')
axes[0].set_title('Accuracy Before LoRA')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Accuracy')

axes[1].bar(categories, final_category_accuracy.numpy(), color='green')
axes[1].set_title('Accuracy After LoRA')
axes[1].set_xlabel('Category')
axes[1].set_ylabel('Accuracy')

plt.show()
