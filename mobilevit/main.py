from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from PIL import Image
import requests
from typing import *


def predict(image: Image.Image) -> Dict[str, Any]:
    image_processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
    model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")

    # annotate inputs type: Dict[str, torch.Tensor]
    # pixel_values: torch.Tensor
    # tensor: [batch_size=1, channels=3, height=256, width=256]
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)  # model(pixel_values = inputs["pixel_values"])
    # [1, 1000]
    logits = outputs.logits
    result_distribution = logits.softmax(-1)

    predicted_class_idx = result_distribution.argmax(-1).item()
    print(result_distribution[0][predicted_class_idx])
    return model.config.id2label[predicted_class_idx]


if __name__ == "__main__":
    # 66789
    # testing set can just be urls to images in coco dataset
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    prediction = predict(image)
    print(prediction)

"""
1. Load Image
- Load image from coco dataset with selected url given by dataset group.
    - training_set.json (train our own local model), test_set.json (calculate success rate) 
- Images are covered to PIL Image type.

2. Convert Image to PyTorch Tensor
- image_processor

3. Construct a predict function with input as PyTorch Tensor and output as {label: str, confidence: float}

4a. WB
- Perturbate images with WB algorithm with predict function.
    - add perturbation variable to model input: model(norm(input_tensor + delta))
    - use different methods to optimize delta
- calculate success rate

4b. BB1 & BB2
- Train local models with BB algorithm with predict function.
- replace the "model" from above with local models

https://huggingface.co/datasets/imagenet-1k/tree/main/data
"""
