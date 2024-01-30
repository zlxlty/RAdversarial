import torch
import torch.nn as nn
import numpy as np
import copy

from .. import AttackMethod

## https://github.com/cmhcbb/attackbox/blob/65a82f8ea6beedc1b4339aa05b08443d5c489b8a/sign_sgd/foolbox/attacks/localsearch.py#L122
class LocSearchAdv(AttackMethod):
    """A black-box attack based on the idea of greedy local search.

    This implementation is based on the algorithm in [1]_.

    References
    ----------
    .. [1] Nina Narodytska, Shiva Prasad Kasiviswanathan, "Simple
           Black-Box Adversarial Perturbations for Deep Networks",
           https://arxiv.org/abs/1612.06299

    """

    def do_perturbation(self,input_tensor, true_label_idx,
                 r=1.7, p=20., d=5, t=5, R=150):

        """A black-box attack based on the idea of greedy local search.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, correctly classified image. If image is a
            numpy array, label must be passed as well. If image is
            an :class:`Adversarial` instance, label must not be passed.
        label : int
            The reference label of the original image. Must be passed
            if image is a numpy array, must not be passed if image is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial image, otherwise returns
            the Adversarial object.
        r : float
            Perturbation parameter that controls the cyclic perturbation;
            must be in [0, 2]
        p : float
            Perturbation parameter that controls the pixel sensitivity
            estimation
        d : int
            The half side length of the neighborhood square
        t : int
            The number of pixels perturbed at each round
        R : int
            An upper bound on the number of iterations

        """

        a = input_tensor
        del input_tensor

        # TODO: incorporate the modifications mentioned in the manuscript
        # under "Implementing Algorithm LocSearchAdv"

        assert 0 <= r <= 2
        
        print(r, p ,d ,t, R, sep="  ")

        def normalize(im):
            min_, max_ = torch.min(a), torch.max(a)

            im = im - (min_ + max_) / 2
            im = im / (max_ - min_)

            LB = -1 / 2
            UB = 1 / 2
            return im, LB, UB

        def unnormalize(im):
            min_, max_ = torch.min(a), torch.max(a)

            im = im * (max_ - min_)
            im = im + (min_ + max_) / 2
            return im

        Im = a
        Im, LB, UB = normalize(Im)

        (_, color_channel, w, h) = a.shape

        def random_locations():
            n = int(0.1 * h * w)
            locations = np.random.RandomState().permutation(h * w)[:n]
            p_x = locations % w
            p_y = locations // w
            pxy = list(zip(p_x, p_y))
            pxy = np.array(pxy)
            return pxy

        def pert(Ii, p, x, y):
            Im = copy.deepcopy(Ii)
            sign = torch.sign((Im[:, :, x, y]))
            Im[:, :, x, y] = p * sign
            return Im

        def cyclic(r, Ibxy):
            result = r * Ibxy
            if result < LB:
                result = result + (UB - LB)
            elif result > UB:
                result = result - (UB - LB)
            assert LB <= result <= UB
            return result

        def top_k_prediction_prob(pred, k):
            prob = nn.Softmax(dim=1)(pred)
            _, ind = prob.sort(descending= True)
            return ind[0][:k]

        Ii = Im
        PxPy = random_locations()

        for iter in range(R):
            print(iter)
            # Computing the function g using the neighborhood
            # IMPORTANT: random subset for efficiency
            L = [pert(Ii, p, x, y) for x, y in PxPy]

            def score(Its):
                scores = []
                Its = torch.stack(Its)
                for it in Its:
                    it = unnormalize(it)
                    pred = self.model.predict(it)
                    score = nn.Softmax(dim=1)(pred)[0, true_label_idx].item()
                    scores.append(score)
                return scores

            scores = score(L)

            indices = np.argsort(scores)[:t]
            
            scores.sort()
            scores = scores[:t]
            print(sum(scores) / len(scores))

            PxPy_star = PxPy[indices]

            # Generation of new perturbed image Ii
            for x, y in PxPy_star:
                for b in range(color_channel):
                    Ii[:, b, x, y] = cyclic(r, Ii[:, b, x, y]) 
                    
            pred_I_hat = self.model.predict(unnormalize(Ii))

            pred_max_class = pred_I_hat.max(dim=1)[1].item()
            print("Predicted class: ", self.model.id2label(pred_max_class))
            print("Predicted probability:", nn.Softmax(dim=1)(pred_I_hat)[0, pred_max_class].item())
            
            indexes = top_k_prediction_prob(pred_I_hat, 5)
            print(indexes)
            if(true_label_idx not in indexes):
                return self

            # Update a neighborhood of pixel locations for the next round
            PxPy = [
                (x, y)
                for _a, _b in PxPy_star
                for x in range(_a - d, _a + d + 1)
                for y in range(_b - d, _b + d + 1)]
            PxPy = [(x, y) for x, y in PxPy if 0 <= x < w and 0 <= y < h]
            PxPy = list(set(PxPy))
            PxPy = np.array(PxPy)