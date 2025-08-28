from turtle import forward
from unittest import result
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gender_labels=['Male', 'Female']

class FaceGender(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
        
    def forward(self, x):
        return self.backbone(x)
    
class FaceAge(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        
    def forward(self, x):
        return self.backbone(x)
    
class Fas(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        self.backbone.classifier[3] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.classifier[3].in_features, 2)
        )
        
    def forward(self, x):
        return self.backbone(x)
   
  
def load_models(age_path='ModelAge.pth', gender_path='ModelGender.pth', fas_path='ModelFAS.pth'):
    age_model = FaceAge().to(device)
    age_model.load_state_dict(torch.load(age_path, map_location=device))
    age_model.eval()

    gender_model = FaceGender().to(device)
    gender_model.load_state_dict(torch.load(gender_path, map_location=device))
    gender_model.eval()
    
    fas_model = Fas().to(device)
    fas_model.load_state_dict(torch.load(fas_path, map_location=device))
    fas_model.eval()

    return age_model, gender_model, fas_model


def predict(image: Image.Image, age_model, gender_model):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        age_pred = max(0, int(age_model(img_tensor).item()))
        gender_logits = gender_model(img_tensor)
        gender_pred = torch.argmax(gender_logits, dim=1).item()

    return age_pred, gender_labels[gender_pred]

def fas_face(image: Image.Image, fas_model):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = fas_model(img_tensor)
        result = torch.argmax(logits, dim=1).item()
    return result
