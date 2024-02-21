from PIL import Image
import torch
import os

from defines import IMAGE_PATH, EVAL_PATH, CONFIG_PATH, DATASET_PATH
from classifiers import get_target_model, label2id
from attacks import PGDMethod, FGSMMethod, NoMethod, LocSearchAdv


# values are standard normalization for ImageNet images, 
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
def generate_image_data(skip=0):
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
    skipped = 0
    for image_name in os.listdir(image_folder):
        if skipped < skip:
            skipped += 1
            continue
        image = Image.open(f"{image_folder}/{image_name}")
        true_label = name2label[image_name].split("\n")[0]
        yield image_name, image, true_label


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

attack_methods = {
    "LocSearchAdv_new": {
        "config": f"{CONFIG_PATH}/locsearchadv_new.yaml",
        "method": LocSearchAdv
    },
    "LocSearchAdv_old": {
        "config": f"{CONFIG_PATH}/locsearchadv_old.yaml",
        "method": LocSearchAdv
    },
}

'''
Choose the target models and attack methods here.
'''
TARGET_MODEL = [
    "MobileViT", 
    # "Surrogate", 
    # "ResNet50"
]
METHOD_NAMES = [
    "LocSearchAdv",
    "FGSM",
    "PGD",
    "NO"
]

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    TARGETxMETHOD = [(target, method) for target in TARGET_MODEL for method in METHOD_NAMES]
        
    for model_name, method_name in TARGETxMETHOD:
        target_model = get_target_model(model_name, device)
        true_target_model = None
        if model_name == "Surrogate":
            true_target_model = get_target_model("MobileViT", device)
        
        attack = attack_methods[method_name]
        config_path = attack["config"]
        image_data_generator = generate_image_data()
        
        img_dir = f"{IMAGE_PATH}/NonImageNet/{method_name}_{model_name}"
        eval_dir = f"{EVAL_PATH}/NonImageNet/{method_name}_{model_name}"
        create_dir(img_dir)
        create_dir(eval_dir)
        image_processed = 0
        
        for image_name, original_image, true_label in image_data_generator:
            print(image_name, true_label, sep= "   ")
            
            input_tensor = target_model.preprocess(original_image)
            true_label_idx = label2id(true_label)

            attack["method"](target_model, config_path)\
                .do_perturbation(input_tensor, true_label_idx)\
                .do_eval(input_tensor, true_label_idx, 5, true_target_model)\
                .save_eval_to_json(image_name, true_label_idx, f"{eval_dir}/{method_name}_{model_name}.json")\
                .save_perturbation_to_png(f"{img_dir}/perturbed_{image_name}.png")\

            image_processed += 1
            print(f"Image processed: {image_processed}")

