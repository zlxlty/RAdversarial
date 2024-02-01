from classifiers import get_target_model
from attacks import PGDMethod, LocSearchAdv
from PIL import Image
import requests
import torch
import torch.nn as nn

import os
from defines import IMAGE_PATH, EVAL_PATH


# values are standard normalization for ImageNet images, 
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
def generate_image_data():
    '''
    TODO[dataset]: Get image data from ImageNet1k folder
    Returns:
        image: PIL.Image
        true_label_idx: int
    '''
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # This is a fake image getter for testing
    # yield "pig.jpg", Image.open(f"{IMAGE_PATH}/pig.jpg"), 341
    yield "cat.jpg", Image.open(requests.get(url, stream=True).raw), 282
    yield "milo.jpg",  Image.open(f"{IMAGE_PATH}/milo.jpg"), 281
    yield "ox.jpg", Image.open(f"{IMAGE_PATH}/ox.jpg"), 345

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

FGSMMethod = None
attack_methods = {
    # "PGD": {
    #     "config": "attacks/config/pgd.yaml",
    #     "method": PGDMethod
    # },
    # "FGSM": {
    #     "epsilon": 5./255.,
    #     "num_iter": 1,
    #     "targeted": False,
    #     "method": FGSMMethod
    # }
    "LocSearchAdv": {
        "config": "attacks/config/locsearchadv.yaml",
        "method": LocSearchAdv
    }
}

if __name__ == '__main__':
    # testing set can just be urls to images in coco dataset
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(f"{IMAGE_PATH}/ox.jpg")
    # pig_img = Image.open("./images/pig.jpg") # opening a image 
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    target_model = get_target_model("MobileViT", device)

    # max_class = logit.max(dim=1)[1].item()
    # print(max_class)
    # print("Predicted class: ", target_model.id2label(max_class))
    # print("Predicted probability:", nn.Softmax(dim=1)(logit)[0, max_class].item())
    
    for method_name in attack_methods:
        attack = attack_methods[method_name]
        config_path = attack["config"]
        image_data_generator = generate_image_data()
        
        img_directory = f"{IMAGE_PATH}/{method_name}"
        create_dir(img_directory)
            
        for image_name, original_image, true_label_idx in image_data_generator:
            print(image_name, true_label_idx, sep= "   ")
            input_tensor = target_model.preprocess(original_image)
            
            attack["method"](target_model, config_path)\
                .do_perturbation(input_tensor, true_label_idx)\
                .do_eval(true_label_idx)\
                .save_perturbation_to_png(f"{img_directory}/perturbed_{image_name[:-4]}.png")\
                .save_eval_to_json(image_name, true_label_idx, f"{EVAL_PATH}/{method_name}.json")
            
            print()
    