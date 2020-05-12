#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:42:56 2020

@author: alex
"""

import torch
import torch.nn as nn
import numpy as np
from NQS_pytorch import Op, Psi

# system parameters
b=0.5   # b-field strength
J=1     # nearest neighbor interaction strength
L = 3   # system size

# Define operators to use in the Hamiltonian
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.array([[1, 0], [0, -1]])

szsz = np.kron(sigmaz, sigmaz)

# initiate the operators and the matrix they are fed
nn_interaction=Op(-J*szsz)
b_field=Op(b*sigmax)


for i in range(L):  # Specify the sites upon which the operators act
    # specify the arbitrary sites which the operators will act on
    b_field.add_site([i])
    nn_interaction.add_site([i,(i+1)%L])

'''##### Define Neural Networks and initialization funcs for psi  #####'''

def psi_init(L, H=2*L, Form='euler'):
    toy_model=nn.Sequential(nn.Linear(L,H),nn.Sigmoid(), 
                       nn.Linear(H,1),nn.Sigmoid()) 
    H2=round(H/2)
    toy_model2=nn.Sequential(nn.Linear(L,H2),nn.Sigmoid(),
                     nn.Linear(H2,1),nn.Sigmoid()) 

    ppsi=Psi(toy_model,toy_model2, L, form=Form)
    
    return ppsi

'''##### Checking the calculation of O_omega (dln(psi_omega)/domega) #####'''

N_samples=100
s=np.random.randint(-1,high=1,size=[N_samples,L]); s[s==0]=1; 
s=torch.tensor(s,dtype=torch.float)
L=3
H=5*L

ppsi_mod=psi_init(L,H,'euler') # without mult, initializes params randomly

psi0=ppsi_mod.complex_out(s) # the original psi

outi=ppsi_mod.imag_comp(s)

# what we calculated the gradients should be
outi.mean().backward()

pars=list(ppsi_mod.imag_comp.parameters())
grad0=pars[0].grad 
dw=0.01 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    pars[0][0][0]=pars[0][0][0]+dw
 
psi1=ppsi_mod.complex_out(s)
if N_samples==1:
    diff=np.log(psi1)-np.log(psi0)
else:
    diff=np.mean(np.log(psi1))-np.mean(np.log(psi0))
deriv_i=(np.imag(diff))/dw
# not:
# deriv_r=(np.angle(diff))/dw
# This is because the log(psi) breaks the peices into: 
# log(Psi)=log(psi_1)+1j*psi_2 such that:
# d/dw Im(log(Psi)) = d/dw psi_2
# and d/dw Re(log(Psi))=d/dw psi_1

print('numberical deriv: ', deriv_i, '\n pytorch deriv: ', grad0[0][0].item(), \
        '\n ratio: ', deriv_i/grad0[0][0].item() )

ppsi_mod=psi_init(L,H,'euler') # without mult, initializes params randomly

psi0=ppsi_mod.complex_out(s) # the original psi

outr=ppsi_mod.real_comp(s)

# what we calculated the gradients should be
outr.log().mean().backward()

pars=list(ppsi_mod.real_comp.parameters())
grad0=pars[0].grad 
dw=0.01 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    pars[0][0][0]=pars[0][0][0]+dw
 
psi1=ppsi_mod.complex_out(s)
if N_samples==1:
    diff=np.log(psi1)-np.log(psi0)
else:
    diff=np.mean(np.log(psi1))-np.mean(np.log(psi0))
deriv_r=(np.real(diff))/dw

print('numberical deriv: ', deriv_r.item(), '\n pytorch deriv: ', grad0[0][0].item(), \
        '\n ratio: ', deriv_r.item()/grad0[0][0].item() )

''' Now with vector version '''

N_samples=100
s=np.random.randint(-1,high=1,size=[N_samples,L]); s[s==0]=1; 
s=torch.tensor(s,dtype=torch.float)
L=3
H=4*L

ppsi_vec=psi_init(L,H,'vector') # without mult, initializes params randomly

psi0=ppsi_vec.complex_out(s) # the original psi

outr=ppsi_vec.real_comp(s)

pars=list(ppsi_vec.real_comp.parameters())

# have to make a sort of copy for the pars in numpy as torch will not use complex #'s
par_list=[]
for k in range(len(pars)):
    par_list.append(np.zeros_like(pars[k].detach().numpy(), dtype=complex))
    
# what we calculated the gradients should be
for n in range(N_samples):
    
    ppsi_vec.real_comp.zero_grad()
    outr[n].backward(retain_graph=True) # retain so that buffers aren't cleared 
                                        # and it can be applied again
    m= 1/psi0[n]  # 1/Psi(s) multiplier according to derivative
            
    # unfortunately, the backwards call adds the gradient, 
    # making each sequential grad call a combination of previous. Must be 
    # separated by using zero_grad
    
    for kk in range(len(pars)):
        par_list[kk]+=(pars[kk].grad.detach().numpy())*m
  
    
# manually do the mean
for kk in range(len(pars)):
    par_list[kk]=par_list[kk]/N_samples

grad0=par_list[0]
dw=0.01 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    pars[0][0][0]=pars[0][0][0]+dw
 
psi1=ppsi_vec.complex_out(s)
if N_samples==1:
    diff=np.log(psi1)-np.log(psi0)
else:
    diff=np.mean(np.log(psi1))-np.mean(np.log(psi0))
deriv=diff/dw

print('numberical deriv: ', deriv, '\n pytorch deriv: ', grad0[0][0] , \
        '\n ratio: ', deriv/grad0[0][0] )

# finally for complex vec i
ppsi_vec=psi_init(L,H,'vector') # without mult, initializes params randomly

psi0=ppsi_vec.complex_out(s) # the original psi

out=ppsi_vec.imag_comp(s)

pars=list(ppsi_vec.imag_comp.parameters())

par_list=[]
for k in range(len(pars)):
    par_list.append(np.zeros_like(pars[k].detach().numpy(), dtype=complex))
    
# what we calculated the gradients should be
for n in range(N_samples):
    
    ppsi_vec.imag_comp.zero_grad()
    out[n].backward(retain_graph=True) 
    
    m= 1j/psi0[n]  # 1/Psi(s) multiplier according to derivative
               
    for kk in range(len(pars)):
        par_list[kk]+=(pars[kk].grad.detach().numpy())*m
  
# manually do the mean
for kk in range(len(pars)):
    par_list[kk]=par_list[kk]/N_samples

grad0=par_list[0]
dw=0.01 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    pars[0][0][0]=pars[0][0][0]+dw
 
psi1=ppsi_vec.complex_out(s)
if N_samples==1:
    diff=np.log(psi1)-np.log(psi0)
else:
    diff=np.mean(np.log(psi1))-np.mean(np.log(psi0))
deriv_i=diff/dw

print('numberical deriv: ', deriv_i, '\n pytorch deriv: ', grad0[0][0] , \
        '\n ratio: ', deriv_i/grad0[0][0] )




