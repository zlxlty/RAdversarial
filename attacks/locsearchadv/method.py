import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import torchvision.transforms as transforms
import math

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

    def pert(self, I, p, pts_to_pert):
        img = copy.deepcopy(I)
        for x, y in pts_to_pert:
            sign = torch.sign((img[:, :, x, y]))
            img[:, :, x, y] = p * sign
        return img
    
    def get_pic_coordinates(self, grid_x, grid_y, grid_size, x_dim, y_dim):
        start_x = grid_x * grid_size
        start_y = grid_y * grid_size
        org_pts = [
            (start_x+i,start_y+j) 
            for i in range (grid_size) 
            for j in range (grid_size)
        ]
        
        org_pts = [ 
            (x, y) for x, y in org_pts if 0 <= x < x_dim and 0 <= y < y_dim
        ]
        return org_pts

    def do_perturbation(self, input_tensor, true_label_idx) -> AttackMethod:
        ## Get hyperparam
        p = self.param_config["p"]
        r = self.param_config["r"]
        d = self.param_config["d"]
        t = self.param_config["t"]
        k = self.param_config["k"]
        R = self.param_config["R"]
        grid_size = self.param_config["grid_size"]
        
        iters_to_ignore = self.param_config["iters_to_ignore"]
        
        self.init_picked_percentage = self.param_config["init_percentage"]
        self.LB = self.param_config["LB"]
        self.UB = self.param_config["UB"]
        
        MODEL_LB = torch.min(input_tensor)
        MODEL_UB = torch.max(input_tensor)
        I = self.rescale(input_tensor, MODEL_LB, MODEL_UB, self.LB, self.UB)
        (_, color_channel, x_dim, y_dim) = I.shape
        
        grid_dim = math.ceil(x_dim / grid_size)
        num_pixel = int(grid_dim*grid_dim*self.init_picked_percentage)
        
        ## Randomly pick pixels to start
        P_X, P_Y = np.random.choice(range(grid_dim), num_pixel), np.random.choice(range(grid_dim), num_pixel)
        # copying the image
        I_hat = copy.deepcopy(I)
        
        pts_perturbed = []
        
        for iter in range (R):
            if iter // (iters_to_ignore + 1) != 0:
                pts_perturbed = pts_perturbed[t:]
            print("\nIter: {}    Num Pixels Ignored: {}".format(iter, len(pts_perturbed)))
            self.number_iteration = iter
            ## Compute function g
            scores = []
            for i in range(len(P_X)):
                pts_to_pert = self.get_pic_coordinates(P_X[i], P_Y[i], grid_size, x_dim, y_dim)    
                img = self.pert(I_hat, p, pts_to_pert)
                img = self.rescale(img, self.LB, self.UB, MODEL_LB, MODEL_UB)

                pred = self.model.predict(img)
                score = nn.Softmax(dim=1)(pred)[0, true_label_idx].item()
                scores.append(score)
            
            sorted_scores = np.argsort(scores)
            scores.sort()
            first_t_scores = scores[:t]
            P_XI, P_YI = P_X[sorted_scores], P_Y[sorted_scores]
            avg = sum(first_t_scores) / len(first_t_scores)
            
            if avg >= 0.5:
                p *= 1.1
            elif avg <= 0.1:
                p *= 0.9
            
            print(avg, p, sep= "   ")
            
            num_perturbed = 0
            i = 0
            P_X_perturbed, P_Y_perturbed = [], []
            while num_perturbed < t:  
                if i == len(P_XI):
                    break
                x, y = P_XI[i], P_YI[i]
                if ((x, y) in pts_perturbed):
                    i += 1
                    continue
                
                pts_perturbed.append((x, y))    
                P_X_perturbed.append(x)
                P_Y_perturbed.append(y)
                pts_to_pert = self.get_pic_coordinates(x, y, grid_size, x_dim, y_dim)    
                for x, y in pts_to_pert:
                    for j in range (color_channel):
                        I_hat[:, j, x, y] = self.cyclic(I_hat, r, j, x, y) 
                
                num_perturbed += 1
                i += 1
            
            
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
            for i in range (num_perturbed):
                x, y = P_X_perturbed[i], P_Y_perturbed[i]                
                for row in range(-d, d+1):
                    for col in range(-d, d+1):
                        new_x = x + col
                        new_y = y + row
                        if (self.inRange(new_x, 0, grid_dim) and self.inRange(new_y, 0, grid_dim)):
                            P_X.append(new_x)
                            P_Y.append(new_y)
            P_X = np.array(P_X)    
            P_Y = np.array(P_Y)   
            
            if len(P_X) == len(P_Y) == 0:
                break

        
        return self
