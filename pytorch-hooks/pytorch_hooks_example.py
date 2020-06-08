# author: Vipul Vaibhaw
# Use this code for educational purpose.

import torch

# we will start with a simple example 
# let's learn to register hooks

x = torch.randn(1,1)
w = torch.randn(1,1, requires_grad=True) # necessary to enable requires_grad because hooks work on gradients

w.register_hook(lambda x:print (x))





