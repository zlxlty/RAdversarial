from classifiers import get_target_model
from attacks import PGDMethod, FGSMMethod, LocSearchAdv
from PIL import Image
import requests
import torch
import torch.nn as nn

import os
from defines import IMAGE_PATH, EVAL_PATH, CONFIG_PATH, DATASET_PATH

NUM_IMAGES_TO_TEST = 1

# values are standard normalization for ImageNet images, 
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
def generate_image_data():
    '''
    TODO[dataset]: Get image data from ImageNet1k folder
    Returns:
        image: PIL.Image
        true_label_idx: int
    '''
    image_folder = f"{DATASET_PATH}/images"
    label_txt = f"{DATASET_PATH}/labels.txt"
    with open(label_txt, "r") as f:
        name2label = {line.split(": ")[0]: line.split(": ")[1] for line in f.readlines()}
    # iterate and open each image file in image folder
    for image_name in os.listdir(image_folder):
        image = Image.open(f"{image_folder}/{image_name}")
        true_label = name2label[image_name].split("\n")[0]
        yield image_name, image, true_label


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

attack_methods = {
    "LocSearchAdv": {
        "config": f"{CONFIG_PATH}/locsearchadv.yaml",
        "method": LocSearchAdv
    },
    # "PGD": {
    #     "config": f"{CONFIG_PATH}/pgd.yaml",
    #     "method": PGDMethod
    # },
    # "FGSM": {
    #     "config": f"{CONFIG_PATH}/fgsm.yaml",
    #     "method": FGSMMethod
    # }
}

if __name__ == '__main__':    
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    target_model = get_target_model("MobileViT", device)

    # max_class = logit.max(dim=1)[1].item()
    # print(max_class)
    # print("Predicted class: ", target_model.id2label(max_class))
    # print("Predicted probability:", nn.Softmax(dim=1)(logit)[0, max_class].item())
    
    for method_name in attack_methods:
        attack = attack_methods[method_name]
        config_path = attack["config"]
        
        img_dir = f"{IMAGE_PATH}/{method_name}"
        eval_dir = f"{EVAL_PATH}/{method_name}"
        create_dir(img_dir)
        create_dir(eval_dir)
            
        image_data_generator = generate_image_data()
        
        for image_name, original_image, true_label in image_data_generator:
            print(image_name, true_label, sep= "   ")
            
            input_tensor = target_model.preprocess(original_image)
            true_label_idx = target_model.label2id(true_label)
            
            attack["method"](target_model, config_path)\
                .do_perturbation(input_tensor, true_label_idx)\
                .do_eval(input_tensor ,true_label_idx, topk=3)\
                .save_perturbation_to_png(f"{img_dir}/perturbed_{image_name}.png")\
                .save_eval_to_json(image_name, true_label_idx, f"{eval_dir}/{method_name}_exp.json")
            
            print()
    