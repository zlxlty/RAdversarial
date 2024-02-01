import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image



def do_perturbation(input_tensor, label_idx, model, epsilon=2.0/255.0):
    input_tensor.requires_grad = True
    
    # Forward pass the data through the model
    # The model() function is expected to return a tuple of (batch_size = 1, logits)
    output = model.predict(input_tensor)
    # Apply log_softmax to the output to get the log-probabilities
    output = F.log_softmax(output[None,:], dim=1).to(input_tensor.device)
    
    # Convert label_idx to a tensor and ensure it's on the same device as input_tensor
    # label_idx_tensor = torch.LongTensor([label_idx]).to(input_tensor.device)
    label_idx_tensor = torch.nn.functional.one_hot(torch.tensor([label_idx]), num_classes=1000).to(input_tensor.device)

    # Calculate the loss
    loss = F.nll_loss(output, label_idx_tensor)
    
    # Zero all existing gradients
    model.model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = input_tensor.grad.data

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = input_tensor + epsilon*sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    pred = model.predict(perturbed_image)

    print("True class probability:", nn.Softmax(dim=1)(pred)[0, label_idx].item())
    max_class = pred.max(dim=1)[1].item()
    print("Predicted class: ", model.id2label(max_class))
    print("Predicted probability:", nn.Softmax(dim=1)(pred)[0, max_class].item())
    
    # Convert BGR to RGB
    perturbed_image = perturbed_image.squeeze().detach().cpu()  # Remove the batch dimension and move to CPU
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(((perturbed_image)*255).detach().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)

    # Convert the numpy array to a PIL image
    pil_image = Image.fromarray(rgb_image.astype('uint8'))

    # Save the image as "final_image.jpg"
    pil_image.save("perturbed_image_fgsm.jpg")


    # return perturbed_image