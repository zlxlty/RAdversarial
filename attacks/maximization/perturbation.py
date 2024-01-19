import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import utils
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import cv2
from copy import deepcopy
       
def do_perturbation(input_tensor, label_idx, model, epsilon=2.0/255.0, num_iteration = 30):
    delta = torch.zeros_like(input_tensor, requires_grad=True)
    
    opt= optim.SGD([delta], lr=1e-1)
    for t in range(num_iteration):
        pred = model.predict(input_tensor + delta)
        loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([label_idx]))
        if t % 5 == 0:
            print(t, loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        delta.data.clamp_(-epsilon, epsilon)
    
    
    print("True class probability:", nn.Softmax(dim=1)(pred)[0, label_idx].item())
    max_class = pred.max(dim=1)[1].item()
    print("Predicted class: ", model.id2label(max_class))
    print("Predicted probability:", nn.Softmax(dim=1)(pred)[0, max_class].item())
    
    # Convert BGR to RGB
    final_image = input_tensor + delta
    final_image = final_image.squeeze().detach().cpu()  # Remove the batch dimension and move to CPU
    
    # print(final_image)
    # Convert BGR to RGB
    print(final_image)
    rgb_image = cv2.cvtColor(((final_image)*255).detach().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)

    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(rgb_image.astype('uint8'))

    # Save the image as "final_image.jpg"
    pil_image.save("final_image.jpg")
    
    final_image = final_image.squeeze().detach().cpu()  # Remove the batch dimension and move to CPU
    # final_image = transforms.ToPILImage()(final_image)  # Convert the tensor to a PIL image
    # final_image.save("final_image.jpg")  # Save the image as a JPEG file
    
    # color_channels = final_image[0]
    # index = torch.Tensor([2, 1, 0]).long()
    # color_channels[index] = color_channels.clone()
    # final_image = color_channels[None, :, :, :]
    # utils.save_image(final_image, "./images/perturbated_pig.jpg")