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
from NQS_pytorch import Op, Psi, O_local, kron_matrix_gen
import itertools

# system parameters
b=0.5   # b-field strength
J=1     # nearest neighbor interaction strength
L = 6   # system size
burn_in=1000
N_samples=10000 # number of samples for the Monte Carlo chains

spin=0.5    # routine may not be optimized yet for spin!=0.5
evals=2*np.arange(-spin,spin+1)

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

# this tool creates the full Hamiltonian and is used for energy comparison 
# practical in use of L<=16, otherwise memory requirements can become an issue
op_list=[] 
for ii in range(2): # make sure to change to match Hamiltonian entered above
    op_list.append(sigmaz)
H_szsz=-J*kron_matrix_gen(op_list,len(evals),L,'periodic').toarray()

op_list2=[]
op_list2.append(sigmax)
H_sx=b*kron_matrix_gen(op_list2,len(evals),L,'periodic').toarray()

H_tot=H_szsz+H_sx

s2=np.array(list(itertools.product(evals,repeat=L)))

'''##### Define Neural Networks and initialization funcs for psi  #####'''

#def psi_init(L, H=2*L, Form='euler'):
#    toy_model=nn.Sequential(nn.Linear(L,H),nn.Sigmoid(), 
#                       nn.Linear(H,1))#,nn.Sigmoid()) 
#    H2=round(H/2)
#    toy_model2=nn.Sequential(nn.Linear(L,H2),nn.Sigmoid(),
#                     nn.Linear(H2,1))#,nn.Sigmoid()) 
#
#    ppsi=Psi(toy_model,toy_model2, L, form=Form)
#    
#    return ppsi

# A simple as possible network
def psi_init(L, H=2*L, Form='euler'):
    H=1
    H2=1
    toy_model=nn.Sequential(nn.Linear(L,H))#
#            ,nn.Sigmoid(), nn.Linear(H,1),nn.Sigmoid()) 
#    H2=round(H/2)
    toy_model2=nn.Sequential(nn.Linear(L,H2))
#    ,nn.Sigmoid(), nn.Linear(H2,1),nn.Sigmoid()) 

    ppsi=Psi(toy_model,toy_model2, L, form=Form)
    
    return ppsi

''' ########## Expand the O_omega routines to calculate grad of E ##########'''

''' ## ANGLE ## '''

H=2*L
ppsi_mod=psi_init(L,H,'euler') # without mult, initializes params randomly

## HAVE TO GENERATE THE SAMPLES VIA MONTE CARLO! Otherwise may not be wieghted correctly ##
sb=ppsi_mod.sample_MH(burn_in,spin=0.5)
sn=ppsi_mod.sample_MH(N_samples,spin=0.5, s0=sb[-1])
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
#analytic=0
#for ii in range(N_samples):
#    analytic=analytic+2*np.imag(E_loc[ii]*sn[ii])/N_samples
#print('Pytorch derivative vs analytical derivative (only works when '\
# 'when using single layer affine map): \n', analytic, '\n', np.real(grad0),'\n')

dw=0.01 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    modi_params[0][0][0]=modi_params[0][0][0]+dw

# uncomment to have burn ins and resampling
#sb=angle_net.sample_MH(burn_in,spin=0.5)
#sn=angle_net.sample_MH(N_samples,spin=0.5,s0=sb[-1])
#s=torch.tensor(sn,dtype=torch.float)
    
# recalculate the energy
[H_nn2, H_b2]=O_local(nn_interaction,s.numpy(),angle_net),O_local(b_field,s.numpy(),angle_net)
new_energy=np.sum(np.mean(H_nn2+H_b2,axis=0)) 
deriv=np.real((new_energy-energy)/dw)

print('ANGLE: \n numberical deriv: ', deriv, '\n pytorch deriv: ',  \
      grad0[0][0].item(), '\n ratio : ' , deriv.item()/grad0[0][0].item() ,\
        '\n relative error: ', np.abs((grad0[0][0].item()-deriv)/deriv) )

wvf0=ppsi_mod.complex_out(torch.tensor(s2,dtype=torch.float))
wvf1=angle_net.complex_out(torch.tensor(s2,dtype=torch.float))
E_tot0=np.matmul(np.matmul(np.conjugate(wvf0.T),H_tot),wvf0)/(np.matmul(np.conjugate(wvf0.T),wvf0))
E_tot1=np.matmul(np.matmul(np.conjugate(wvf1.T),H_tot),wvf1)/(np.matmul(np.conjugate(wvf1.T),wvf1))
dif=(E_tot1-E_tot0)/dw

print('\n DE/dw with matrix and wavefunction difference: ', dif, ' \n pytorch derivative'\
' relative error with respect to wvf calculated E: ', np.abs(grad0[0][0].item()-dif)/dif,\
'\n with ratio: ', grad0[0][0].item()/dif, '\n\n')

''' ## MODULUS ## '''

ppsi_mod=psi_init(L,H,'euler') # without mult, initializes params randomly
mod_net=copy.deepcopy(ppsi_mod)
sb=ppsi_mod.sample_MH(burn_in,spin=0.5)
sn=ppsi_mod.sample_MH(N_samples,spin=0.5, s0=sb[-1])
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

#analytic=0
#for ii in range(N_samples):
#    analytic=analytic+2*np.real((np.conj(E_loc[ii])-np.conj(E0)) \
#                                *(sn[ii]/outr[ii].item()))/N_samples
#print('Pytorch derivative vs analytical derivative (only works when '\
# 'when using single layer affine map): \n', analytic, '\n', np.real(grad0),'\n')

pars2=list(mod_net.real_comp.parameters())
dw=0.01 # sometimes less accurate when smaller than 1e-3
#for ii in range(pars2[0].size(1)):
with torch.no_grad():
    pars2[0][0][0]=pars2[0][0][0]+dw
#    pars2[1][0]=pars2[1][0]+dw
 
# MUST RESAMPLE WITH THE CHANGED PSI (but this messes up the comparison!)
#sb=mod_net.sample_MH(burn_in,spin=0.5)
#sn=mod_net.sample_MH(N_samples,spin=0.5,s0=sb[-1])
#s=torch.tensor(sn,dtype=torch.float)

[H_nn, H_b]=O_local(nn_interaction,s.numpy(),mod_net),O_local(b_field,s.numpy(),mod_net)
E_loc=np.sum(H_nn+H_b,axis=1)
E1=np.real(np.mean(E_loc))
deriv_r=np.real((E1-E0)/dw)

print('MODULUS: \n numberical deriv: ', deriv_r.item(), '\n pytorch deriv: ', \
      grad0[0][0].item(), '\n ratio: ', deriv_r.item()/grad0[0][0].item() )

wvf0=ppsi_mod.complex_out(torch.tensor(s2,dtype=torch.float))
wvf1=mod_net.complex_out(torch.tensor(s2,dtype=torch.float))
E_tot0=np.matmul(np.matmul(np.conjugate(wvf0.T),H_tot),wvf0)/(np.matmul(np.conjugate(wvf0.T),wvf0))
E_tot1=np.matmul(np.matmul(np.conjugate(wvf1.T),H_tot),wvf1)/(np.matmul(np.conjugate(wvf1.T),wvf1))
dif=(E_tot1-E_tot0)/dw

print('\n DE/dw with matrix and wavefunction difference: ', dif, ' \n pytorch derivative'\
' relative error with respect to wvf calculated E: ', np.abs(grad0[0][0].item()-dif)/dif,\
'\n with ratio: ', grad0[0][0].item()/dif, '\n\n')

''' Now with vector version '''

''' ## REAL COMP ## '''

ppsi_vec=psi_init(L,H,'vector')  # without mult, initializes params randomly
real_net=copy.deepcopy(ppsi_vec)

sb=ppsi_vec.sample_MH(burn_in,spin=0.5)
sn=ppsi_vec.sample_MH(N_samples,spin=0.5, s0=sb[-1])
s=torch.tensor(sn,dtype=torch.float)

psi0=ppsi_vec.complex_out(s)

[H_nn, H_b]=O_local(nn_interaction,s.numpy(),ppsi_vec),O_local(b_field,s.numpy(),ppsi_vec)
E_loc=np.sum(H_nn+H_b,axis=1)
E0=np.real(np.mean(E_loc))

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

#analytic=0
#for ii in range(N_samples):
#    analytic=analytic+2*np.real((np.conj(E_loc[ii])-np.conj(E0)) \
#                    *(sn[ii]/psi0[ii].item()))/N_samples
#print('Pytorch derivative vs analytical derivative (only works when '\
# 'when using single layer affine map): \n', analytic, '\n', np.real(grad0),'\n')

dw=0.01 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    pars[0][0][0]=pars[0][0][0]+dw
 
#sb=ppsi_vec.sample_MH(burn_in,spin=0.5)
#sn=ppsi_vec.sample_MH(N_samples,spin=0.5,s0=sb[-1])
#s=torch.tensor(sn,dtype=torch.float)
    
[H_nn, H_b]=O_local(nn_interaction,s.numpy(),ppsi_vec),O_local(b_field,s.numpy(),ppsi_vec)
E_loc=np.sum(H_nn+H_b,axis=1)
E1=np.real(np.mean(E_loc) )

deriv=(E1-E0)/dw

print('REAL COMP: \n numberical deriv: ', deriv, '\n pytorch deriv: ',  \
        grad0[0][0].item() , '\n ratio: ', deriv/grad0[0][0].item() )

wvf0=real_net.complex_out(torch.tensor(s2,dtype=torch.float))
wvf1=ppsi_vec.complex_out(torch.tensor(s2,dtype=torch.float))
E_tot0=np.matmul(np.matmul(np.conjugate(wvf0.T),H_tot),wvf0)/(np.matmul(np.conjugate(wvf0.T),wvf0))
E_tot1=np.matmul(np.matmul(np.conjugate(wvf1.T),H_tot),wvf1)/(np.matmul(np.conjugate(wvf1.T),wvf1))
dif=(E_tot1-E_tot0)/dw

print('\n DE/dw with matrix and wavefunction difference: ', dif, ' \n pytorch derivative'\
' relative error with respect to wvf calculated E: ', np.abs(grad0[0][0].item()-dif)/dif,\
'\n with ratio: ', grad0[0][0].item()/dif, '\n\n')

''' ## IMAG COMP ## '''
# finally for complex vec i
ppsi_vec=psi_init(L,H,'vector') # without mult, initializes params randomly
imag_net=copy.deepcopy(ppsi_vec)

sb=ppsi_vec.sample_MH(burn_in,spin=0.5)
sn=ppsi_vec.sample_MH(N_samples,spin=0.5, s0=sb[-1])
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

#analytic=0
#for ii in range(N_samples):
#    analytic=analytic+2*np.real(1j*(np.conj(E_loc[ii])-np.conj(E0)) \
#                    *(sn[ii]/psi0[ii].item()))/N_samples
#print('Pytorch derivative vs analytical derivative (only works when '\
# 'when using single layer affine map): \n', analytic, '\n', np.real(grad0),'\n')

dw=0.01 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    pars[0][0][0]=pars[0][0][0]+dw
 
#sb=ppsi_vec.sample_MH(burn_in,spin=0.5)
#sn=ppsi_vec.sample_MH(N_samples,spin=0.5, s0=sb[-1])
#s=torch.tensor(sn,dtype=torch.float)
    
[H_nn, H_b]=O_local(nn_interaction,s.numpy(),ppsi_vec),O_local(b_field,s.numpy(),ppsi_vec)
E_loc=np.sum(H_nn+H_b,axis=1)
E1=np.mean(E_loc) 
deriv_i=(E1-E0)/dw

print('IMAG COMP: \n numberical deriv: ', deriv_i, '\n pytorch deriv: ',  \
        grad0[0][0].item() , '\n ratio: ', deriv_i/grad0[0][0].item() )

wvf0=imag_net.complex_out(torch.tensor(s2,dtype=torch.float))
wvf1=ppsi_vec.complex_out(torch.tensor(s2,dtype=torch.float))
E_tot0=np.matmul(np.matmul(np.conjugate(wvf0.T),H_tot),wvf0)/(np.matmul(np.conjugate(wvf0.T),wvf0))
E_tot1=np.matmul(np.matmul(np.conjugate(wvf1.T),H_tot),wvf1)/(np.matmul(np.conjugate(wvf1.T),wvf1))
dif=(E_tot1-E_tot0)/dw

print('\n DE/dw with matrix and wavefunction difference: ', dif, ' \n pytorch derivative'\
' relative error with respect to wvf calculated E: ', np.abs(grad0[0][0].item()-dif)/dif,\
'\n with ratio: ', grad0[0][0].item()/dif, '\n\n')