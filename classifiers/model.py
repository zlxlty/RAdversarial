from typing import Dict
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from PIL import Image
import torch
import torch.optim as optim
from torchvision import transforms, models
import requests
from torchvision import transforms
from typing import *
from .utils import id2label, label2id

class TargetModel():
    def __init__(self, device: str):
        self.device = device
        
    def getDevice(self) -> str:
        return self.device
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pass
    
    def normalize(self, x, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return transforms.Normalize(mean, std)(x)
    
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
        outputs = self.model(self.normalize(
            inputs, 
            torch.Tensor([0.485, 0.456, 0.406]), 
            torch.Tensor([0.229, 0.224, 0.225])
        ))
        return outputs
    
class SurrogateModel(TargetModel):
    def __init__(self, device: str):
        super().__init__(device)
        self.model = torch.load("./classifiers/pretrained_model/surrogate_35.pth").to(device)
        self.load_label_dict()
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0).to(self.device)
    
    def load_label_dict(self) -> int:
        f = open("./classifiers/pretrained_model/label_ids.txt", "r")
        lines = f.readlines()
        f.close()
        self.label_dict = {}
        for line in lines:
            g = line.strip().split(": ")
            id_num = int(g[0])
            label = g[1]
            self.label_dict[id_num] = label
                
    def predict(self, inputs: torch.Tensor) -> Dict[str, Any]:
        outputs = self.model(self.normalize(
            inputs,
            torch.Tensor([0.5, 0.5, 0.5]),
            torch.Tensor([0.5, 0.5, 0.5])
        ))
        
        remapped_outputs = outputs.clone()
        for i in range(999):
            remapped_outputs[0][label2id(self.label_dict[i])] = outputs[0][i]
        
        
        return remapped_outputs
    
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
    elif model_name == "ResNet50":
        model = Resnet50Model(device)
    elif model_name == "Surrogate":
        model = SurrogateModel(device)
    else:
        raise ValueError(f"model_name {model_name} not supported")
    
    model.model.eval()
    return model