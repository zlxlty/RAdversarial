from PIL import Image
import torch
import os
import yaml
import copy
import json

from defines import IMAGE_PATH, EVAL_PATH, CONFIG_PATH, DATASET_PATH
from classifiers import get_target_model, label2id, VGG16
from attacks import PGDMethod, FGSMMethod, LocSearchAdv, NoMethod


def generate_image_data(example_img = None, skip=0, json_images = False):
    """
    [dataset]: Get image data from ImageNet1k folder
    Returns:
        image: PIL.Image
        true_label_idx: int
    """
    image_folder = f"{DATASET_PATH}/images"
    label_txt = f"{DATASET_PATH}/labels.txt"
    with open(label_txt, "r") as f:
        name2label = {
            line.split(": ")[0]: line.split(": ")[1] for line in f.readlines()
        }

    if example_img:
        image = Image.open(f"./images/{example_img['name']}")
        yield example_img, image, example_img["true_label"]
    elif json_images:
        with open(f'LocSearchAdv_MobileVit.json') as f:
            images = json.load(f)
            for image in images:
                image_name = image["input_name"]
                image = Image.open(f"{image_folder}/{image_name}")
                true_label = name2label[image_name].split("\n")[0]
                yield image_name, image, true_label
    else:
        # iterate and open each image file in image folder
        skipped = 0
        for image_name in os.listdir(image_folder)[:50]:
            if skipped < skip:
                skipped += 1
                continue
            image = Image.open(f"{image_folder}/{image_name}")
            true_label = name2label[image_name].split("\n")[0]
            yield image_name, image, true_label


attack_methods = {
    # "LocSearchAdv_neighbor": {
    #     "config": f"{CONFIG_PATH}/LocSearchExp/locsearchadv_d.yaml",
    #     "method": LocSearchAdv
    # },
    "LocSearchAdv_pixel_per_round": {
        "config": f"{CONFIG_PATH}/LocSearchExp/locsearchadv_t.yaml",
        "method": LocSearchAdv
    },
   
   "LocSearchAdv_r": {
        "config": f"{CONFIG_PATH}/LocSearchExp/locsearchadv_r.yaml",
        "method": LocSearchAdv
    },
}

"""
Choose the target models and attack methods here.
"""
TARGET_MODEL = [
    "MobileViT", 
    # "Surrogate", 
    # "ResNet50"
]
METHOD_NAMES = [
    "LocSearchAdv",
    # "LocSearchAdv_old",
    # "FGSM",
    # "PGD",
    # "NO"
]

exp = ["d", "t", "r"]

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model_name = "MobileViT"
    
    for index, method_name in enumerate(attack_methods):
        target_model = get_target_model(model_name, device)
        true_target_model = None

        attack = attack_methods[method_name]
        config_path = attack["config"]
        
        img_dir = f"{IMAGE_PATH}/paper_exp/{method_name}"
        eval_dir = f"{EVAL_PATH}/paper_exp/{method_name}"
        
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)
            
        exp_param = config_dict[exp[index]]
        print()
        print(exp_param)

        # Panda: ILSVRC2012_val_00006543_n02510455.JPEG
        image_processed = 0
        for cur_param in exp_param:
            print(exp[index], cur_param)
            cur_exp = copy.deepcopy(config_dict)
            cur_exp[exp[index]] = cur_param
            for image_name, original_image, true_label in generate_image_data():
                print(image_name, true_label, sep= "   ")
                
                input_tensor = target_model.preprocess(original_image)
                true_label_idx = label2id(true_label)
                attack["method"](target_model, cur_exp).do_perturbation(
                    input_tensor, true_label_idx
                )\
                .do_eval(
                    input_tensor, true_label_idx, 5, true_target_model
                )\
                .save_eval_to_json(
                    image_name,
                    true_label_idx,
                    f"{eval_dir}/{method_name}_{model_name}_{exp[index]}:{cur_param}.json",
                )
                # .save_perturbation_to_png(f"{img_dir}/perturbed_{image_name}.png")

                image_processed += 1
                print(f"Image processed: {image_processed}\n")
