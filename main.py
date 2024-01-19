from classifiers import get_target_model
from attacks import do_perturbation
from PIL import Image
import requests
import torch.nn as nn

# values are standard normalization for ImageNet images, 
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py

if __name__ == '__main__':
    # testing set can just be urls to images in coco dataset
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    pig_img = Image.open("./images/final_image.jpg")
    
    target_model = get_target_model("MobileViT")
    input_tensor = target_model.preprocess(pig_img)
    print(input_tensor)
    
    # do_perturbation(input_tensor, 341, target_model)
    
    logit = target_model.predict(input_tensor)
    max_class = logit.max(dim=1)[1].item()
    print("Predicted class: ", target_model.id2label(max_class))
    print("Predicted probability:", nn.Softmax(dim=1)(logit)[0, max_class].item())
    