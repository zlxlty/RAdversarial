## Cost function f is the prob of the perturbed image to true class
## eg. model thinks the perturbed pig img's prob as pig

import torch
import torch.nn as nn
import numpy as np
import copy
import torchvision.transforms as transforms

init_picked_percentage = 0.1
LB = -1
UB = 1

def rescale(I, min, max, UB, LB):
    return (I - min) * (UB - LB) / (max - min) + LB

def cyclic(r, b, x, y):
    return

def pert(I, p, x, y):
    img = copy.deepcopy(I)
    sign = np.sign(img[:, :, x, y])
    img[:, :, x, y] = p * sign
    return img

## I should be a (batch, color, x_dim, y_dim) tensor
def do_locsearchadv(I, p, r, model):
    I = rescale(I, 0, 1, UB, LB)
    (_, _, x_dim, y_dim) = I.shape

    num_pixel = int(x_dim*y_dim*init_picked_percentage)
    P_X, P_Y = np.random.choice(range(x_dim), num_pixel), np.random.choice(range(y_dim), num_pixel)
    
    print(P_X, P_Y, sep="\n")
    init_perturbed = []
    score = []
    # for i in range(num_pixel):
    #     img = pert(I, p, P_X[i], P_Y[i])
    #     init_perturbed.append(img)
        
    #     pred = model.predict(img)
    #     nn.Softmax(dim=1)(pred)[0, max_class].item()
    
    img = pert(I, p, 0, 0)
    init_perturbed.append(img)
    
    img_to_pred = copy.deepcopy(img)
    img_to_pred = rescale(img_to_pred, LB, UB, 1, 0)
    
    pred = model.predict(img_to_pred)
    max_class = pred.max(dim=1)[1].item()
    print("Predicted class: ", model.id2label(max_class))
    print("Predicted probability:", nn.Softmax(dim=1)(pred)[0, max_class].item())
    
    final_image = img_to_pred.squeeze().detach().cpu()  # Remove the batch dimension and move to CPU
    final_image = transforms.ToPILImage()(final_image)  # Convert the tensor to a PIL image
    final_image.save("1pixel_image.jpg")  # Save the image as a JPEG file
    
    I = rescale(I, LB, UB, 1, 0)
    pred = model.predict(I)
    max_class = pred.max(dim=1)[1].item()
    print("Predicted class: ", model.id2label(max_class))
    print("Predicted probability:", nn.Softmax(dim=1)(pred)[0, max_class].item())
    final_image = I.squeeze().detach().cpu()  # Remove the batch dimension and move to CPU
    final_image = transforms.ToPILImage()(final_image)  # Convert the tensor to a PIL image
    final_image.save("2pixel_image.jpg")  # Save the image as a JPEG file
    
    return I
# input = rescale(input)

# (_, color_channel, x_dim, y_dim) = input.shape

# num_pixel = int(x_dim*y_dim*init_picked_percentage)
# P_X, P_Y = np.random.choice(range(x_dim), num_pixel), np.random.choice(range(y_dim), num_pixel)

# print(P_X, P_Y, sep="\n")

# for (x, y) in (P_X, P_Y):
#     print(x, y)

# # print(no_batch[0])
# print(input[:, :, 0, 0])
# i = pert(input, 2, 0, 0)
# print(i[:, :, 0, 0])
# # print(input[:, :, 0, 0])