from classifiers import get_target_model
from attacks import PGDMethod, FGSMMethod, LocSearchAdv
from PIL import Image
import requests
import torch
import torch.nn as nn

import os
from defines import IMAGE_PATH, EVAL_PATH, CONFIG_PATH, BENCHMARK_PATH

NUM_IMAGES_TO_TEST = 1

# values are standard normalization for ImageNet images, 
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
def generate_image_data(model):
    '''
    TODO[dataset]: Get image data from ImageNet1k folder
    Returns:
        image: PIL.Image
        true_label_idx: int
    '''
    with open(f"{BENCHMARK_PATH}/labels.txt", 'r') as f:
        for line in f.readlines()[:NUM_IMAGES_TO_TEST]:
            [file_name, label] = line.strip().split(": ")
            img = Image.open(f"{BENCHMARK_PATH}/images/{file_name}")
            img_name = label.replace(", ", "&").replace(" ", "_")
            id = model.label2id(label)
            yield img_name, img, id

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

attack_methods = {
    "LocSearchAdv": {
        "config": "attacks/config/locsearchadv.yaml",
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
    # testing set can just be urls to images in coco dataset
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(f"{IMAGE_PATH}/ox.jpg")
    # pig_img = Image.open("./images/pig.jpg") # opening a image 
    
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
        image_data_generator = generate_image_data(target_model)
        
        img_dir = f"{IMAGE_PATH}/{method_name}"
        eval_dir = f"{EVAL_PATH}/{method_name}"
        create_dir(img_dir)
        create_dir(eval_dir)
            
        for image_name, original_image, true_label_idx in image_data_generator:
            print(image_name, true_label_idx, sep= "   ")
            input_tensor = target_model.preprocess(original_image)
            
            attack["method"](target_model, config_path)\
                .do_perturbation(input_tensor, true_label_idx)\
                .do_eval(input_tensor ,true_label_idx, topk=3)\
                .save_perturbation_to_png(f"{img_dir}/perturbed_{image_name}.png")\
                .save_eval_to_json(image_name, true_label_idx, f"{eval_dir}/{method_name}_exp.json")
            
            print()
    