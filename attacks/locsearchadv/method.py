import torch
import torch.nn as nn
import numpy as np
import copy
import torchvision.transforms as transforms

from .. import AttackMethod


## Adaptive p
class LocSearchAdv(AttackMethod):

    init_picked_percentage = 0.1
    LB = -1 #This is just a guess for what the lower and upper bounds should be
    UB = 1  #This is just a guess for what the lower and upper bounds should be

    def cyclic(self, I, r, b, x, y): 
        """ r is the perturbation parameter"""
        
        specific_data = I[:, b, x, y] * r # this is not called correctly
        if(specific_data < self.LB):
            return specific_data + (self.UB - self.LB)
        elif(specific_data > self.UB):
            return specific_data - (self.UB - self.LB)
        else: 
            return specific_data 

    def rescale(self, I, min, max, LB, UB):
        I_copy = copy.deepcopy(I)
        return (I_copy - min) * (UB - LB) / (max - min) + LB

    def inRange(self, val, lower_bound, upper_bound):
        return lower_bound <= val < upper_bound        

    def top_k_prediction_prob(self, pred, k):
        prob = nn.Softmax(dim=1)(pred)
        top, ind = prob.sort(descending= True)
        print(top[0][0])
        return ind[0][:k]

    def pert(self, I, p, x, y):
        img = copy.deepcopy(I)
        sign = torch.sign((img[:, :, x, y]))
        img[:, :, x, y] = p * sign
        return img

    ## I should be a (batch, color, x_dim, y_dim) tensor
    def do_perturbation(self, input_tensor, true_label_idx):
        ## Get hyperparam
        p = self.param_config["p"]
        r = self.param_config["r"]
        d = self.param_config["d"]
        t = self.param_config["t"]
        k = self.param_config["k"]
        R = self.param_config["R"]
        self.init_picked_percentage = self.param_config["init_percentage"]
        self.LB = self.param_config["LB"]
        self.UB = self.param_config["UB"]
        
        MODEL_LB = torch.min(input_tensor)
        MODEL_UB = torch.max(input_tensor)
        I = self.rescale(input_tensor, MODEL_LB, MODEL_UB, self.LB, self.UB)
        (_, color_channel, x_dim, y_dim) = I.shape
        num_pixel = int(x_dim*y_dim*self.init_picked_percentage)
        ## Randomly pick pixels to start
        P_X, P_Y = np.random.choice(range(x_dim), num_pixel), np.random.choice(range(y_dim), num_pixel)
        
        I_hat = copy.deepcopy(I) # copying the image
        
        for iter in range (R):
            print(iter)
            ## Compute function g
            scores = []
            for i in range(len(P_X)):    
                img = self.pert(I_hat, p, P_X[i], P_Y[i])
                img = self.rescale(img, self.LB, self.UB, MODEL_LB, MODEL_UB)
                pred = self.model.predict(img)
                score = nn.Softmax(dim=1)(pred)[0, true_label_idx].item()
                scores.append(score)
            
            sorted_scores = np.argsort(scores)
            scores.sort()
            scores = scores[:t]
            P_XI, P_YI = (P_X[sorted_scores])[:t], (P_Y[sorted_scores])[:t]
            
            print(sum(scores) / len(scores))
            ## Generate perturb image I_hat
            #want to traverse px i and py i 
            ## Check if I_hat is an adversaria image
            #need to copy the image 

            for i in range (t) : 
                for j in range (color_channel):
                    I_hat[:, j, P_XI[i] ,P_YI[i]] = self.cyclic(I_hat, r, j, P_XI[i] ,P_YI[i]) 
            #predict with I-hat

            img_I_hat = self.rescale(I_hat, self.LB, self.UB, MODEL_LB, MODEL_UB)
            pred_I_hat = self.model.predict(img_I_hat)
            
            self.logit = pred_I_hat
            self.perturbed_input = img_I_hat
            
            indexes = self.top_k_prediction_prob(pred_I_hat, k)
            print(indexes)
            if(true_label_idx not in indexes):
                return self
            
            ## Update neighborhood of pixel location for next round
            P_X, P_Y = [], []
            
            ## Need optimization
            for i in range (t):
                x, y = P_XI[i], P_YI[i]
                for row in range(-d, d+1):
                    for col in range(-d, d+1):
                        new_x = x + col
                        new_y = y + row
                        if (self.inRange(new_x, 0, x_dim) and self.inRange(new_y, 0, y_dim)):
                            P_X.append(new_x)
                            P_Y.append(new_y)
            P_X = np.array(P_X)    
            P_Y = np.array(P_Y)   

        
        return self
