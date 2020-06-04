#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:43:25 2020

@author: alex
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from NQS_pytorch import Op, Psi, O_local

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

#def psi_init(L, H=2*L, Form='euler'):
#    toy_model=nn.Sequential(nn.Linear(L,H),nn.Sigmoid(), 
#                       nn.Linear(H,1),nn.Sigmoid()) 
#    H2=round(H/2)
#    toy_model2=nn.Sequential(nn.Linear(L,H2),nn.Sigmoid(),
#                     nn.Linear(H2,1),nn.Sigmoid()) 
#
#    ppsi=Psi(toy_model,toy_model2, L, form=Form)
#    
#    return ppsi

# A simple as possible network
def psi_init(L, H=2*L, Form='euler'):
    H=1
    H2=1
    toy_model=nn.Sequential(nn.Linear(L,H))
#                            nn.Sigmoid(), nn.Linear(H,1),nn.Sigmoid()) 
#    H2=round(H/2)
    toy_model2=nn.Sequential(nn.Linear(L,H2))
#    ,nn.Sigmoid(), nn.Linear(H2,1),nn.Sigmoid()) 

    ppsi=Psi(toy_model,toy_model2, L, form=Form)
    
    return ppsi

''' ########## Expand the O_omega routines to calculate grad of E ##########'''

''' ## ANGLE ## '''
N_samples=100

L=3
H=2*L
ppsi_mod=psi_init(L,H,'euler') # without mult, initializes params randomly

## HAVE TO GENERATE THE SAMPLES VIA MONTE CARLO! Otherwise may not be wieghted correctly ##
sn=ppsi_mod.sample_MH(N_samples,spin=0.5)
s=torch.tensor(sn,dtype=torch.float)
modi_params=list(ppsi_mod.imag_comp.parameters())

[H_nn, H_b]=O_local(nn_interaction,s.numpy(),ppsi_mod),O_local(b_field,s.numpy(),ppsi_mod)
energy=np.sum(np.mean(H_nn+H_b,axis=0)) 
E_loc=np.sum(H_nn+H_b,axis=1)

angle_net=copy.deepcopy(ppsi_mod)

outi=ppsi_mod.imag_comp(s)

# what we calculated the gradients should be
mult=torch.tensor(2*np.imag(-np.conj(E_loc)),dtype=torch.float)

ppsi_mod.imag_comp.zero_grad()
(outi*mult[:,None]).mean().backward()

modi_params=list(angle_net.imag_comp.parameters())
pars=list(ppsi_mod.imag_comp.parameters())
grad0=pars[0].grad 

# This is the analytical term for the derivative of a single layer affine map (matches)
analytic=0
for ii in range(N_samples):
    analytic=analytic+2*np.imag(E_loc[ii]*sn[ii])/N_samples
print('Pytorch derivative" ', grad0, '\n vs analytical derivative (only works when) '\
      'when using single layer affine map: ', analytic)

dw=0.01 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    modi_params[0][0][0]=modi_params[0][0][0]+dw
 
#sn=ppsi_mod.sample_MH(N_samples,spin=0.5)
#s=torch.tensor(sn,dtype=torch.float)
    
# recalculate the energy
[H_nn2, H_b2]=O_local(nn_interaction,s.numpy(),angle_net),O_local(b_field,s.numpy(),angle_net)
new_energy=np.sum(np.mean(H_nn2+H_b2,axis=0)) 
deriv=np.real((new_energy-energy)/dw)
#deriv=E0*( (np.angle(E1)-np.angle(E0))/dw)
#deriv=np.imag(E1-E0)/dw

print('numberical deriv: ', deriv, '\n pytorch deriv: ', grad0[0][0].item(), \
      '\n ratio : ' , deriv.item()/grad0[0][0].item() ,\
        '\n relative error: ', np.abs((grad0[0][0].item()-deriv)/deriv) )

''' ## MODULUS ## '''

ppsi_mod=psi_init(L,H,'euler') # without mult, initializes params randomly
mod_net=copy.deepcopy(ppsi_mod)
sn=ppsi_mod.sample_MH(N_samples,spin=0.5)
s=torch.tensor(sn,dtype=torch.float)

[H_nn, H_b]=O_local(nn_interaction,sn,ppsi_mod),O_local(b_field,sn,ppsi_mod)
E_loc=np.sum(H_nn+H_b,axis=1)
E0=np.real(np.mean(E_loc))

outr=ppsi_mod.real_comp(s)

#mult=torch.tensor(2*np.real(E_loc-E0))
mult=torch.tensor((2*np.real(np.conj(E_loc)-np.conj(E0))))#/(outr.detach().numpy()).T).T)

# what we calculated the gradients should be
(outr.log()*mult[:,None]).mean().backward()

pars=list(ppsi_mod.real_comp.parameters())
grad0=pars[0].grad 

analytic=0
for ii in range(N_samples):
    analytic=analytic+2*np.real((np.conj(E_loc[ii])-np.conj(E0)) \
                                *(sn[ii]/outr[ii].item()))/N_samples
print('Pytorch derivative ', grad0, '\n vs analytical derivative (only works when) '\
      'when using single layer affine map: ', analytic)

pars2=list(mod_net.real_comp.parameters())
dw=0.001 # sometimes less accurate when smaller than 1e-3
#for ii in range(pars2[0].size(1)):
with torch.no_grad():
    pars2[0][0][0]=pars2[0][0][0]+dw
#    pars2[1][0]=pars2[1][0]+dw
 
# MUST RESAMPLE WITH THE CHANGED PSI (but this messes up the comparison!)
#sn=mod_net.sample_MH(N_samples,spin=0.5)
#s=torch.tensor(sn,dtype=torch.float)

[H_nn, H_b]=O_local(nn_interaction,s.numpy(),mod_net),O_local(b_field,s.numpy(),mod_net)
E_loc=np.sum(H_nn+H_b,axis=1)
E1=np.real(np.mean(E_loc))
deriv_r=np.real((E1-E0)/dw)

print('numberical deriv: ', deriv_r.item(), '\n pytorch deriv: ', grad0[0][0].item(), \
        '\n ratio: ', deriv_r.item()/grad0[0][0].item() )

''' Now with vector version '''

''' ## REAL COMP ## '''

N_samples=1000
L=3
H=2*L

ppsi_vec=psi_init(L,H,'vector')  # without mult, initializes params randomly

sn=ppsi_vec.sample_MH(N_samples,spin=0.5)
s=torch.tensor(sn,dtype=torch.float)

psi0=ppsi_vec.complex_out(s)

[H_nn, H_b]=O_local(nn_interaction,s.numpy(),ppsi_vec),O_local(b_field,s.numpy(),ppsi_vec)
E_loc=np.sum(H_nn+H_b,axis=1)
E0=np.mean(E_loc)

outr=ppsi_vec.real_comp(s)

ppsi_vec.real_comp.zero_grad()
pars=list(ppsi_vec.real_comp.parameters())

# have to make a sort of copy to record the modified gradient
grad_list=copy.deepcopy(pars)
with torch.no_grad():
    for param in grad_list:
        param.copy_(torch.zeros_like(param))
        param.requires_grad=False
    
# what we calculated the gradients should be
for n in range(N_samples):
    
    ppsi_vec.real_comp.zero_grad()
    outr[n].backward(retain_graph=True) # retain so that buffers aren't cleared 
                                        # and it can be applied again
    m= 2*np.real((np.conj(E_loc[n])-np.conj(E0))/psi0[n])  # 1/Psi(s) multiplier according to derivative
          
    # way 1
#    for kk in range(len(pars)):
#        par_list[kk]+=(pars[kk].grad.detach().numpy())*m
  
    # way 2
    m=torch.tensor(m,dtype=torch.float)
    for kk in range(len(pars)):
        with torch.no_grad():
            grad_list[kk]+=(pars[kk].grad)*(m/N_samples)
#        pars[kk].grad=pars[kk].grad*(m/N_samples)
    
# manually do the mean
#for kk in range(len(pars)):
#    par_list[kk]=par_list[kk]/N_samples

for kk in range(len(pars)):
    pars[kk].grad=grad_list[kk]

grad0=grad_list[0]

analytic=0
for ii in range(N_samples):
    analytic=analytic+2*np.real((np.conj(E_loc[ii])-np.conj(E0)) \
                    *(sn[ii]/psi0[ii].item()))/N_samples
print('Pytorch derivative ', grad0, '\n vs analytical derivative (only works when) '\
      'when using single layer affine map: ', analytic)

dw=0.01 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    pars[0][0][0]=pars[0][0][0]+dw
 
[H_nn, H_b]=O_local(nn_interaction,s.numpy(),ppsi_vec),O_local(b_field,s.numpy(),ppsi_vec)
E_loc=np.sum(H_nn+H_b,axis=1)
E1=np.real(np.mean(E_loc) )

deriv=(E1-E0)/dw

print('numberical deriv: ', deriv, '\n pytorch deriv: ', grad0[0][0].item() , \
        '\n ratio: ', deriv/grad0[0][0].item() )

''' ## IMAG COMP ## '''
# finally for complex vec i
ppsi_vec=psi_init(L,H,'vector') # without mult, initializes params randomly

sn=ppsi_vec.sample_MH(N_samples,spin=0.5)
s=torch.tensor(sn,dtype=torch.float)

psi0=ppsi_vec.complex_out(s) # the original psi

[H_nn, H_b]=O_local(nn_interaction,s.numpy(),ppsi_vec),O_local(b_field,s.numpy(),ppsi_vec)
E_loc=np.sum(H_nn+H_b,axis=1)
E0=np.mean(E_loc) 

out=ppsi_vec.imag_comp(s)

pars=list(ppsi_vec.imag_comp.parameters())

par_list=[]
for k in range(len(pars)):
    par_list.append(np.zeros_like(pars[k].detach().numpy(), dtype=complex))

# what we calculated the gradients should be
for n in range(N_samples):

    ppsi_vec.imag_comp.zero_grad()    
    out[n].backward(retain_graph=True) # retain so that buffers aren't cleared 
                                        # and it can be applied agai
    with torch.no_grad():        
        m= 2*np.real((np.conj(E_loc[n])-np.conj(E0))*1j/psi0[n]) # 1i/Psi according to derivative
            
    for kk in range(len(pars)):
        par_list[kk]+=(pars[kk].grad.detach().numpy())*m            
            
# manually do the mean
for kk in range(len(pars)):
    par_list[kk]=par_list[kk]/N_samples

grad0=par_list[0]

analytic=0
for ii in range(N_samples):
    analytic=analytic+2*np.real(1j*(np.conj(E_loc[ii])-np.conj(E0)) \
                    *(sn[ii]/psi0[ii].item()))/N_samples
print('Pytorch derivative ', grad0, '\n vs analytical derivative (only works when) '\
      'when using single layer affine map: ', analytic)

dw=0.01 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    pars[0][0][0]=pars[0][0][0]+dw
 
[H_nn, H_b]=O_local(nn_interaction,s.numpy(),ppsi_vec),O_local(b_field,s.numpy(),ppsi_vec)
E_loc=np.sum(H_nn+H_b,axis=1)
E1=np.mean(E_loc) 
deriv_i=(E1-E0)/dw

print('numberical deriv: ', deriv_i, '\n pytorch deriv: ', grad0[0][0].item() , \
        '\n ratio: ', deriv_i/grad0[0][0].item() )

