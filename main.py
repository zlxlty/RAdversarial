from classifiers import get_target_model, label2id, VGG16
from attacks import PGDMethod, FGSMMethod, NoMethod
from PIL import Image
import torch
import os

from defines import IMAGE_PATH, EVAL_PATH, CONFIG_PATH, DATASET_PATH
from classifiers import get_target_model, label2id
from attacks import PGDMethod, FGSMMethod, LocSearchAdv, NoMethod


# values are standard normalization for ImageNet images, 
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
def generate_image_data():
    '''
    [dataset]: Get image data from ImageNet1k folder
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
    "FGSM": {
        "config": f"{CONFIG_PATH}/fgsm.yaml",
        "method": FGSMMethod
    },
    "PGD": {
        "config": f"{CONFIG_PATH}/pgd.yaml",
        "method": PGDMethod
    },
    "NO": {
        "config": f"{CONFIG_PATH}/no.yaml",
        "method": NoMethod
    },
}

'''
Choose the target models and attack methods here.
'''
TARGET_MODEL = [
    "MobileViT", 
    # "Surrogate", 
    "ResNet50"
]
METHOD_NAMES = [
    "PGD", 
    "FGSM", 
    # "NO"
]

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    for model_name in TARGET_MODEL:
        target_model = get_target_model(model_name, device)
        true_target_model = None
        if model_name == "Surrogate":
            true_target_model = get_target_model("MobileViT", device)
        
        for method_name in METHOD_NAMES:
            attack = attack_methods[method_name]
            config_path = attack["config"]
            image_data_generator = generate_image_data()
            for image_name, original_image, true_label in image_data_generator:
                input_tensor = target_model.preprocess(original_image)
                true_label_idx = label2id(true_label)
                attack["method"](target_model, config_path)\
                    .do_perturbation(input_tensor, true_label_idx)\
                    .do_eval(input_tensor, true_label_idx, 5, true_target_model)\
                    .save_eval_to_json(image_name, true_label_idx, f"{EVAL_PATH}/{method_name}/{method_name}_{model_name}.json")\
                    .save_perturbation_to_json(f"{IMAGE_PATH}/{method_name}/{model_name}/perturbed_{image_name[:-5]}.json")\
                    .save_perturbation_to_png(f"{IMAGE_PATH}/{method_name}/{model_name}/perturbed_{image_name[:-5]}.png")\
    
