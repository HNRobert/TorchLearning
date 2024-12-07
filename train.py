import pickle
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms

dataset = load_dataset("Bingsu/Cat_and_Dog")

# Define transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply transformations to the datasets
def transform_dataset(examples):
    examples['image'] = [transform(image.convert('RGB')) for image in examples['image']]
    return examples

dataset = dataset.map(transform_dataset, batched=True)
dataset.set_format(type='torch', columns=['image', 'labels'])

# Split the dataset into training and validation sets
train_dataset = dataset['train']
val_dataset = dataset['test']

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load a pre-trained model and modify the final layer
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification (cat vs dog)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50  # Set the number of epochs (big number for enough training)
print("MPS Availability: ", torch.backends.mps.is_available())
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

training_results = {'loss': [], 'accuracy': []}
min_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs = batch['image'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    training_results['loss'].append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

    # Check for early stopping
    if avg_loss < min_loss:
        min_loss = avg_loss
        best_model_state = model.state_dict()  # Save the best model so far
    else:
        print("Training stopped.")
        break  # Stop training if loss starts increasing

# Validation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        inputs = batch['image'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
training_results['accuracy'].append(accuracy)
print(f"Validation Accuracy: {accuracy}%")

# Save the model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), 'best_model.pth')
    print("The best model has been saved as: best_model.pth")

# Save the training results
with open('training_results.pkl', 'wb') as f:
    pickle.dump(training_results, f)