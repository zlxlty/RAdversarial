import torch.optim as optim
import torch
import torch.nn as nn

from .. import AttackMethod

class PGDMethod(AttackMethod):
    def do_perturbation(self, input_tensor, true_label_idx) -> AttackMethod:
        ## Get attack config
        epsilon = self.param_config["epsilon"][0] /  self.param_config["epsilon"][1]
        self.epsilon = self.param_config["epsilon"]
        num_iteration = self.param_config["num_iter"]
        
        torch.manual_seed("2024")
        
        delta = torch.zeros_like(input_tensor, requires_grad=True)

        opt= optim.SGD([delta], lr=1e-1)
        for t in range(num_iteration):
            # normalize
            perturbed_input = torch.clamp(input_tensor + delta, 0, 1)
            logit = self.model.predict(perturbed_input)
            
            label_tensor = torch.LongTensor([true_label_idx]).to(self.model.getDevice())
            loss = -nn.CrossEntropyLoss()(logit, label_tensor)
            if t % 5 == 0:
                print(t, loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            delta.data.clamp_(-epsilon, epsilon)
        
        delta = delta.detach()
        self.perturbed_input = torch.clamp(input_tensor + delta, 0, 1)
        self.logit = self.model.predict(perturbed_input)
        
        return self
        

        