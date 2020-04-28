#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Apr 7 15:13:49 2020

@author: alex
"""

### Start by defining an Artificial Neural Network (ANN) Class ### 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

'''
Simple test module using a built in module/class called nn.Sequential
'''
      
L=20
H=40
N_samples=100

s=np.random.randint(-1,high=1,size=[N_samples,L]); s[s==0]=1; 
s=torch.tensor(s,dtype=torch.float)

# creates an instance of the Sequential class nn.Sigmoid etc usually in forward section
toy_model=nn.Sequential(nn.Linear(L,H),\
                        nn.Sigmoid(),\
                        nn.Linear(H,1),
                        nn.Sigmoid())

out=toy_model(s) # example application

''' we can backprop the gradients with a random loss '''

toy_model.zero_grad()
out.backward(torch.randn(N_samples,1)) # modeling a random loss 

params=list(toy_model.parameters()) # record the parameters
params[2].grad # to access a single grad
with torch.no_grad():
    for param in params:
        param -= 0.1 * param.grad
        
        

class RBM_simple(nn.Module): # inherits from nn.Module
    
    def __init__(self, n_visible, n_hidden):
        # super is a function that creates an instance of the superclass nn.Module 
        # which is the superclass for all Pytorch Neural Networks
        super(RBM_simple,self).__init__()
        self.n_visible=n_visible
        self.n_hidden=n_hidden
        
        
class custom_FFNN(nn.Module):
    def __init__(self):
        super(custom_FFNN,self).__init__()
  



'''
If I define a loss function here using only torch functionals, automatic 
differentiation could potentially be used as normal. Should take op/Hamiltonian 
and a set of samples (or a batch). Use .backward() to apply backprop. 
'''
# def variational_loss(samples, H):
    
