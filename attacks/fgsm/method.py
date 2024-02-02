import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import AttackMethod

class FGSMMethod(AttackMethod):
    def do_perturbation(self, input_tensor, true_label_idx) -> AttackMethod:
        epsilon = self.param_config["epsilon"][0] /  self.param_config["epsilon"][1]
        
        input_tensor.requires_grad = True
        
        # Forward pass the data through the model
        # The model() function is expected to return a tuple of (batch_size = 1, logits)
        logit = self.model.predict(input_tensor)
        # Apply log_softmax to the output to get the log-probabilities        
        # Convert label_idx to a tensor and ensure it's on the same device as input_tensor
        # label_idx_tensor = torch.LongTensor([label_idx]).to(input_tensor.device)
        label_tensor = torch.LongTensor([true_label_idx]).to(self.model.getDevice())

        # Calculate the loss
        loss = -nn.CrossEntropyLoss()(logit, label_tensor)

        # Zero all existing gradients
        self.model.model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = input_tensor.grad.data

        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()

        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = input_tensor + epsilon*sign_data_grad

        # Adding clipping to maintain [0,1] range
        self.perturbed_input = torch.clamp(perturbed_image, 0, 1).detach()        
        self.logit = self.model.predict(perturbed_image).detach()
        
        return self