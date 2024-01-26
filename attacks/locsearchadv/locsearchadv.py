## Cost function f is the prob of the perturbed image to true class
## eg. model thinks the perturbed pig img's prob as pig

import torch
import numpy as np

init_picked_percentage = 0.1
input = torch.load('input_img.pt') # input is an image  so we can 
LB = 0 #This is just a guess for what the lower and upper bounds should be
UP = 1  #This is just a guess for what the lower and upper bounds should be
def cyclic(r, b, x, y): 
    input = torch.load('input_img.pt') #this is just getting an image
    specific_data = r.np([b][x][y]) # this is not called correctly
    if(specific_data < LB):
      
      return 

    #if rI (b,x,y) < LB 
        #return rI(b, x, y) + (UB - LB)
    #elif rI(b,x,y) > UB
        #return rI(b, x, y ) - (UB - LB)
    #else 
        #return rI(b,x,y)
    #return

def pert(I, p, x, y):
    return

## I should be a (batch, color, x_dim, y_dim) tensor
def do_locsearchadv(I, p, r):
    return


input = torch.load('input_img.pt') #loading in an image 

(_, color_channel, x_dim, y_dim) = input.shape

num_pixel = int(x_dim*y_dim*init_picked_percentage)
P_X, P_Y = np.random.choice(range(x_dim), num_pixel), np.random.choice(range(y_dim), num_pixel)

print(P_X, P_Y, sep="\n")
