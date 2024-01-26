import torch
import torch.nn as nn
import numpy as np
import copy
import torchvision.transforms as transforms

init_picked_percentage = 0.1
# input = torch.load('input_img.pt') # input is an image  so we can 
LB = 0 #This is just a guess for what the lower and upper bounds should be
UB = 1  #This is just a guess for what the lower and upper bounds should be
MODEL_LB = 0
MODEL_UB = 1

# np.random.seed(2024)

def cyclic(I, r, b, x, y): 
    """ r is the perturbation parameter"""
    
    specific_data = I[:, b, x, y] * r # this is not called correctly
    
    if(specific_data < LB):
      return specific_data + (UB - LB)
    
    elif(specific_data > UB):
        return specific_data - (UB - LB)
    else: 
        return specific_data 

def rescale(I, min, max, LB, UB):
    return (I - min) * (UB - LB) / (max - min) + LB

def inRange(val, lower_bound, upper_bound):
    val = max(lower_bound, val)
    val = min(val, upper_bound - 1)
    return val           

def top_k_prediction_prob(pred, k):
    prob = nn.Softmax(dim=1)(pred)
    _, ind = prob.sort(descending= True)
    return ind[0][:k]

def pert(I, p, x, y):
    img = copy.deepcopy(I)
    sign = torch.sign((img[:, :, x, y]))
    img[:, :, x, y] = p * sign
    return img

## I should be a (batch, color, x_dim, y_dim) tensor
def do_locsearchadv(I, p, r, d, t, k, R, model):
    ## Init prediction to get the true class
    pred = model.predict(I)
    max_class = pred.max(dim=1)[1].item()
    
    iter = 0 
    I = rescale(I, MODEL_LB, MODEL_UB, LB, UB)
    (_, color_channel, x_dim, y_dim) = I.shape
    num_pixel = int(x_dim*y_dim*init_picked_percentage)
    ## Randomly pick pixels to start
    P_X, P_Y = np.random.choice(range(x_dim), num_pixel), np.random.choice(range(y_dim), num_pixel)
    
    while iter < R:
        print(iter)
        ## Compute function g
        scores = []
        for i in range(len(P_X)):    
            img = pert(I, p, P_X[i], P_Y[i])
            img = rescale(img, LB, UB, MODEL_LB, MODEL_UB)
            pred = model.predict(img)
            score = nn.Softmax(dim=1)(pred)[0, max_class].item()
            scores.append(score)
        sorted_scores = np.argsort(scores)
        P_XI, P_YI = P_X[sorted_scores][:t], P_Y[sorted_scores][:t]
        ## Generate perturb image I_hat
        #want to traverse px i and py i 
        ## Check if I_hat is an adversaria image
        #need to copy the image 
        I_hat = copy.deepcopy(I) # copying the image
        for i in range (t) : 
            for j in range (color_channel):
              I_hat[:, j, P_XI[i] ,P_YI[i]] = cyclic(I_hat, r, j, P_XI[i] ,P_YI[i]) 
        #predict with I-hat

        img_I_hat = rescale(I_hat, LB, UB, MODEL_LB, MODEL_UB)
        pred_I_hat = model.predict(img_I_hat)

        pred_max_class = pred_I_hat.max(dim=1)[1].item()
        print("Predicted class: ", model.id2label(pred_max_class))
        print("Predicted probability:", nn.Softmax(dim=1)(pred_I_hat)[0, pred_max_class].item())
        
        indexes = top_k_prediction_prob(pred_I_hat, k)
        print(indexes)
        if(max_class not in indexes):
            torch.save(img_I_hat, "attacks/locsearchadv/loc_img.pt")
            return True
                    
        ## Update neighborhood of pixel location for next round
        P_X, P_Y = [], []
        
        ## Need optimization
        for i in range (t):
            x, y = P_XI[i], P_YI[i]
            for row in range(-d, d+1):
                for col in range(-d, d+1):
                    P_X.append(inRange(x + col, 0, x_dim))
                    P_Y.append(inRange(y + row, 0, y_dim))
        P_X = np.array(P_X)    
        P_Y = np.array(P_Y)   
        iter += 1

    return False
# input = rescale(input)

# (_, color_channel, x_dim, y_dim) = input.shape

# input = torch.load('input_img.pt') #loading in an image 

# print(P_X, P_Y, sep="\n")

# for (x, y) in (P_X, P_Y):
#     print(x, y)

# # print(no_batch[0])
# print(input[:, :, 0, 0])
# i = pert(input, 2, 0, 0)
# print(i[:, :, 0, 0])
# # print(input[:, :, 0, 0])
