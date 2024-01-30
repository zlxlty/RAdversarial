from classifiers import get_target_model
from attacks import PGDMethod, LocSearchAdv
from PIL import Image
import requests
import torch
import torch.nn as nn
import torch

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
    
    # This is a fake image getter for testing
    yield "pig.jpg", Image.open(f"{IMAGE_PATH}/pig.jpg"), 341

FGSMMethod = None
attack_methods = {
    # "PGD": {
    #     "epsilon": 5./255.,
    #     "num_iter": 40,
    #     "targeted": False,
    #     "method": PGDMethod
    # },
    # "FGSM": {
    #     "epsilon": 5./255.,
    #     "num_iter": 1,
    #     "targeted": False,
    #     "method": FGSMMethod
    # }
    "LocSearchAdv": {
        "p": 8,
        "r": 5, 
        "d": 5, 
        "t": 5, 
        "k": 5, 
        "R": 100,
        "targeted": False,
        "method": LocSearchAdv
    }
}

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    target_model = get_target_model("MobileViT", device)
    
    for method_name in attack_methods:
        attack = attack_methods[method_name]
        image_data_generator = generate_image_data()
        for image_name, original_image, true_label_idx in image_data_generator:
            input_tensor = target_model.preprocess(original_image)
            # .do_perturbation(input_tensor, true_label_idx, attack["epsilon"], attack["num_iter"])\
            attack["method"](target_model)\
                .do_perturbation(input_tensor, true_label_idx, attack["p"], attack["r"], attack["d"],attack["t"], attack["k"], attack["R"])\
                .do_eval(true_label_idx)\
                .save_perturbation_to_png(f"{IMAGE_PATH}/perturbed_{image_name[:-4]}.png")\
                .save_eval_to_json(image_name, true_label_idx, f"{EVAL_PATH}/{method_name}.json")
            
    # logit = target_model.predict(input_tensor)
    # max_class = logit.max(dim=1)[1].item()
    # print("Predicted class: ", target_model.id2label(max_class))
    # print("Predicted probability:", nn.Softmax(dim=1)(logit)[0, max_class].item())
    