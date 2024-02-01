import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import torchvision.transforms as transforms

from .. import AttackMethod


class LocSearchAdv(AttackMethod):

    init_picked_percentage = 0.1
    LB = -1 #This is just a guess for what the lower and upper bounds should be
    UB = 1  #This is just a guess for what the lower and upper bounds should be

    def cyclic(self, I, r, b, x, y): 
        """ r is the perturbation parameter"""
        
        specific_data = I[:, b, x, y] * r
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
        print("Top pred prob: {}".format(top[0][0].item()))
        return ind[0][:k]

    def pert(self, I, p, x, y, grid_size):
        img = copy.deepcopy(I)
        
        start_x = x * grid_size
        start_y = y * grid_size
        pts_to_pert = [(start_x+i,start_y+j) for i in range (grid_size) for j in range (grid_size)]
        
        for x, y in pts_to_pert:
            sign = torch.sign((img[:, :, x, y]))
            img[:, :, x, y] = p * sign
        return img

    def do_perturbation(self, input_tensor, true_label_idx) -> AttackMethod:
        ## Get hyperparam
        p = self.param_config["p"]
        r = self.param_config["r"]
        d = self.param_config["d"]
        t = self.param_config["t"]
        k = self.param_config["k"]
        R = self.param_config["R"]
        grid_size = self.param_config["grid_size"]
        
        self.init_picked_percentage = self.param_config["init_percentage"]
        self.LB = self.param_config["LB"]
        self.UB = self.param_config["UB"]
        
        MODEL_LB = torch.min(input_tensor)
        MODEL_UB = torch.max(input_tensor)
        I = self.rescale(input_tensor, MODEL_LB, MODEL_UB, self.LB, self.UB)
        (_, color_channel, x_dim, y_dim) = I.shape
        
        ## For padding
        assert x_dim == y_dim
        
        padding_val = x_dim % grid_size
        ## Add padding to the image
        padded_image = F.pad(I, (padding_val, padding_val, padding_val, padding_val), mode='constant', value=0)
        
        grid_dim = int((x_dim + 2 * padding_val) / grid_size)
        num_pixel = int(grid_dim*grid_dim*self.init_picked_percentage)
        
        print("Grid size: {}".format(grid_dim))
        
        ## Randomly pick pixels to start
        P_X, P_Y = np.random.choice(range(grid_dim), num_pixel), np.random.choice(range(grid_dim), num_pixel)
        # copying the image
        I_hat = copy.deepcopy(padded_image)
        
        for iter in range (R):
            print(iter)
            ## Compute function g
            scores = []
            for i in range(len(P_X)):    
                img = self.pert(I_hat, p, P_X[i], P_Y[i], grid_size)
                img = self.rescale(img, self.LB, self.UB, MODEL_LB, MODEL_UB)
                img = img[:, :, padding_val: -padding_val , padding_val: -padding_val]

                pred = self.model.predict(img)
                score = nn.Softmax(dim=1)(pred)[0, true_label_idx].item()
                scores.append(score)
            
            sorted_scores = np.argsort(scores)
            scores.sort()
            scores = scores[:t]
            P_XI, P_YI = (P_X[sorted_scores])[:t], (P_Y[sorted_scores])[:t]
            avg = sum(scores) / len(scores)
            
            if avg >= 0.5:
                p *= 1.1
            elif avg <= 0.1:
                p *= 0.9
            
            print(avg, p, sep= "   ")
            
            for i in range (t): 
                start_x = P_XI[i] * grid_size
                start_y = P_YI[i] * grid_size
                pts_to_pert = [(start_x+i,start_y+j) for i in range (grid_size) for j in range (grid_size)]
                
                for x, y in pts_to_pert:
                    for j in range (color_channel):
                        I_hat[:, j, x, y] = self.cyclic(I_hat, r, j, x, y) 

            img_I_hat = self.rescale(I_hat, self.LB, self.UB, MODEL_LB, MODEL_UB)
            img_I_hat = img_I_hat[:, :, padding_val: -padding_val , padding_val: -padding_val]
            pred_I_hat = self.model.predict(img_I_hat)
            
            self.logit = pred_I_hat
            self.perturbed_input = img_I_hat
            
            indexes = self.top_k_prediction_prob(pred_I_hat, k)
            print(indexes)
            if(true_label_idx not in indexes):
                return self
            
            ## Update neighborhood of pixel location for next round
            P_X, P_Y = [], []
            
            for i in range (t):
                x, y = P_XI[i], P_YI[i]
                for row in range(-d, d+1):
                    for col in range(-d, d+1):
                        new_x = x + col
                        new_y = y + row
                        if (self.inRange(new_x, 0, grid_dim) and self.inRange(new_y, 0, grid_dim)):
                            P_X.append(new_x)
                            P_Y.append(new_y)
            P_X = np.array(P_X)    
            P_Y = np.array(P_Y)   

        
        return self
