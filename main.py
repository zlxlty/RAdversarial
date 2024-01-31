from classifiers import get_target_model
from attacks import PGDMethod, LocSearchAdv
from PIL import Image
import requests
import torch
import torch.nn as nn

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
    "PGD": {
        "config": "attacks/config/pgd.yaml",
        "method": PGDMethod
    },
    # "FGSM": {
    #     "epsilon": 5./255.,
    #     "num_iter": 1,
    #     "targeted": False,
    #     "method": FGSMMethod
    # }
    "LocSearchAdv": {
        "p": 18.0,
        "r": 1.5, 
        "d": 10, 
        "t": 10, 
        "k": 5, 
        "R": 150,
        "targeted": False,
        "method": LocSearchAdv
    }
}

if __name__ == '__main__':
    # testing set can just be urls to images in coco dataset
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    pig_img = Image.open("./images/pig.jpg") # opening a image 
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    target_model = get_target_model("ResNet50", device)
    input_tensor = target_model.preprocess(pig_img)
    
    # torch.save(input_tensor, 'input_img.pt')
    # input_tensor = torch.load("attacks/locsearchadv/loc_img.pt")
    # do_locsearchadv(input_tensor, 8, 1.8, 5, 5, 5, 150, target_model)
    # do_perturbation(input_tensor, 341, target_model)
    
    for method_name in attack_methods:
        attack = attack_methods[method_name]
        config_path = attack["config"]
        image_data_generator = generate_image_data()
        for image_name, original_image, true_label_idx in image_data_generator:
            input_tensor = target_model.preprocess(original_image)
            
            attack["method"](target_model, config_path)\
                .do_perturbation(input_tensor, true_label_idx)\
                .do_eval(true_label_idx)\
                .save_perturbation_to_png(f"{IMAGE_PATH}/{method_name}/perturbed_{image_name[:-4]}.png")\
                .save_eval_to_json(image_name, true_label_idx, f"{EVAL_PATH}/{method_name}/{method_name}_exp.json")
            
    # logit = target_model.predict(input_tensor)
    # max_class = logit.max(dim=1)[1].item()
    # print("Predicted class: ", target_model.id2label(max_class))
    # print("Predicted probability:", nn.Softmax(dim=1)(logit)[0, max_class].item())
    