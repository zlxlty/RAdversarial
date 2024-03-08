import torch.optim as optim
import torch
import torch.nn as nn

from .. import AttackMethod

class NoiseMethod(AttackMethod):
    def do_perturbation(self, input_tensor, true_label_idx) -> AttackMethod:
        epsilon = self.param_config["epsilon"][0] /  self.param_config["epsilon"][1]
        
        delta = torch.randn_like(input_tensor, requires_grad=True) * epsilon
        
        perturbed_image = input_tensor + delta

        # Adding clipping to maintain [0,1] range
        self.perturbed_input = torch.clamp(perturbed_image, 0, 1).detach()        
        self.logit = self.model.predict(perturbed_image).detach()
        
        return self
        

        