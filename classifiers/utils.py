import torch
from transformers import MobileViTForImageClassification
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
def id2label(id: int) -> str:
    
    
    return model.config.id2label[id]

def label2id(label: str) -> int:
    
    return model.config.label2id[label]

if __name__ == "__main__":
    tensor1 = torch.Tensor([[6,2,3,4,5]])
    print(tensor1.shape)
    tensor2 = tensor1.clone()
    for i in range(5):
        tensor2[0][i] = tensor1[0][5-i-1]
        
    print(tensor2)
    print(tensor1)
        
        