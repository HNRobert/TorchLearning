import os

import torch
from PIL import Image
from torchvision import models, transforms

# Set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the pre-trained model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

# Define the class names
class_names = ['Cat', 'Dog']

# Predict on test images
for i in range(1, 11):
    image_name = f"CorD{i}.jpg"
    image_path = os.path.join("test_images", image_name)
    if os.path.exists(image_path):
        # Load and predict on test images
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make a prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]
        
        print(f"{image_name}: {predicted_class}")
    else:
        print(f"{image_name} not found.")
