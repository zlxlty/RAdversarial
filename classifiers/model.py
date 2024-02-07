from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from PIL import Image
import torch
import torch.optim as optim
import requests
from typing import *

class TargetModel():
    def __init__(self, device: str):
        pass
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pass
    
    def getDevice(self) -> str:
        pass
    
    def predict(self, inputs: torch.Tensor) -> Dict[str, Any]:
        pass
    
class SurrogateModel(TargetModel):
    def __init__(self, device: str):
        self.model = torch.load("/surrogate_model.pth").to(device)
        pass
    
class MobileViTModel(TargetModel):
    def __init__(self, device: str):
        self.model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small").to(device)
        self.image_processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
        self.device = device
        
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        inputs = self.image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"]
        return inputs.to(self.device)
    
    def id2label(self, id: int) -> str:
        return self.model.config.id2label[id]
    
    def label2id(self, label: str) -> int:
        return self.model.config.label2id[label]

    def getDevice(self) -> str:
        return self.device
    
    def predict(self, inputs: torch.Tensor) -> Dict[str, Any]:
        # annotate inputs type: Dict[str, torch.Tensor]
        # pixel_values: torch.Tensor
        # tensor: [batch_size=1, channels=3, height=256, width=256]
        outputs = self.model(pixel_values=(inputs))  # model(pixel_values = inputs["pixel_values"])
        # [1, 1000]
        
        # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        logits = outputs.logits
        return logits
        # result_distribution = logits.softmax(-1)

        # predicted_class_idx = result_distribution.argmax(-1).item()
        # print(result_distribution[0][predicted_class_idx])
        # return self.model.config.id2label[predicted_class_idx]

    
    
def get_target_model(model_name: str, device: str) -> TargetModel:
    if model_name == "MobileViT":
        model = MobileViTModel(device)
        model.model.eval()
        return model
    else:
        raise ValueError(f"model_name {model_name} not supported")
    
    
# norm(input + delta)
# norm(norm(input) + delta)
# norm(norm(input) + delta)