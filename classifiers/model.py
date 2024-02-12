from typing import Dict
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from PIL import Image
import torch
import torch.optim as optim
from torchvision import transforms, models
import requests
from torchvision import transforms
from typing import *

class TargetModel():
    def __init__(self, device: str):
        self.device = device
        
    def getDevice(self) -> str:
        return self.device
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pass
    
    def predict(self, inputs: torch.Tensor) -> Dict[str, Any]:
        pass
    
    
    
class Resnet50Model(TargetModel):
    def __init__(self, device: str):
        super().__init__(device)
        self.model = models.resnet50(pretrained=True).to(device)
        
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def predict(self, inputs: torch.Tensor) -> Dict[str, Any]:
        def norm(x, mean, std):
            return (x - mean.type_as(x)[None,:,None,None]) / std.type_as(x)[None,:,None,None]

        outputs = self.model(norm(
            inputs, 
            torch.Tensor([0.485, 0.456, 0.406]), 
            torch.Tensor([0.229, 0.224, 0.225])
        ))
        return outputs
    
class SurrogateModel(TargetModel):
    def __init__(self, device: str):
        self.model = torch.load("/surrogate_model.pth").to(device)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        transforms = transforms.Compose([
            transforms.Resize(227),
            transforms.ToTensor(),
        ])
        return transforms(image).unsqueeze(0).to(self.device)
    
    def predict(self, inputs: torch.Tensor) -> Dict[str, Any]:
        return super().predict(inputs)
    
class MobileViTModel(TargetModel):
    def __init__(self, device: str):
        super().__init__(device)
        self.model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small").to(device)
        self.image_processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
        
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        inputs = self.image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"]
        return inputs.to(self.device)    
    
    def predict(self, inputs: torch.Tensor) -> Dict[str, Any]:
        # annotate inputs type: Dict[str, torch.Tensor]
        # pixel_values: torch.Tensor
        # tensor: [batch_size=1, channels=3, height=256, width=256]
        outputs = self.model(pixel_values=(inputs))  # model(pixel_values = inputs["pixel_values"])
        # [1, 1000]
        logits = outputs.logits
        return logits

def get_target_model(model_name: str, device: str) -> TargetModel:
    if model_name == "MobileViT":
        model = MobileViTModel(device)
    elif model_name == "Resnet50":
        model = Resnet50Model(device)
        
    else:
        raise ValueError(f"model_name {model_name} not supported")
    
    model.model.eval()
    return model