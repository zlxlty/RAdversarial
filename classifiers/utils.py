from transformers import MobileViTForImageClassification
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
def id2label(id: int) -> str:
    
    
    return model.config.id2label[id]

def label2id(label: str) -> int:
    
    return model.config.label2id[label]

if __name__ == "__main__":
    print(id2label(230))