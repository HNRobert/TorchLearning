import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms


class CatDogDataset:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.dataset = load_dataset("Bingsu/Cat_and_Dog")
        self.dataset = self.dataset.map(self.transform_dataset, batched=True)
        self.dataset.set_format(type='torch', columns=['image', 'labels'])
        self.train_dataset = self.dataset['train']
        self.val_dataset = self.dataset['test']

    def transform_dataset(self, examples):
        examples['image'] = [self.transform(
            image.convert('RGB')) for image in examples['image']]
        return examples

    def get_dataloaders(self, batch_size=32):
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader


class CatDogModel(models.ResNet):
    def __init__(self):
        super(CatDogModel, self).__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2])
        num_ftrs = self.fc.in_features
        self.fc = nn.Linear(num_ftrs, 2)


def train_model(model, train_loader, device, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

        if avg_loss < min_loss:
            min_loss = avg_loss
            best_model_state = model.state_dict()
        else:
            print("Loss did not decrease, stopping training")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model saved as 'best_model.pth'")


def test_model(model, device):
    model.eval()
    class_names = ['Cat', 'Dog']
    for i in range(1, 11):
        image_name = f"CorD{i}.jpg"
        image_path = f"test_images/{image_name}"
        try:
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                predicted_class = class_names[predicted.item()]
            print(f"{image_name}: {predicted_class}")
        except Exception as e:
            print(f"Error processing {image_name}: {e}")


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    dataset = CatDogDataset()
    train_loader, val_loader = dataset.get_dataloaders()

    model = CatDogModel().to(device)

    # train_model(model, train_loader, device)

    # Load the best model and test it
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    test_model(model, device)


if __name__ == "__main__":
    main()
