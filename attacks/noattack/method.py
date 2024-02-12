from .. import AttackMethod
class NoMethod(AttackMethod):
    def do_perturbation(self, input_tensor, true_label_idx) -> AttackMethod:
        self.perturbed_input = input_tensor
        self.logit = self.model.predict(input_tensor)
        return self