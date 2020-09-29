#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:01:56 2020

@author: alex
"""

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
from NQS_pytorch import Op, Psi, kron_matrix_gen
import itertools
from autograd_hacks_master import autograd_hacks
import time

# system parameters
b=0.5   # b-field strength
J=1     # nearest neighbor interaction strength
L = 3   # system size
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

def psi_init(L, H=2*L, Form='euler'):
    toy_model=nn.Sequential(nn.Linear(L,H),nn.Sigmoid(), 
                       nn.Linear(H,1))#,nn.Sigmoid()) 
    H2=round(H/2)
    toy_model2=nn.Sequential(nn.Linear(L,H2),nn.Sigmoid(),
                     nn.Linear(H2,1))#,nn.Sigmoid()) 

    ppsi=Psi(toy_model,toy_model2, L, form=Form)
    
    return ppsi

# A simple as possible network
#def psi_init(L, H=2*L, Form='euler'):
#    H=1
#    H2=1
#    toy_model=nn.Sequential(nn.Linear(L,H))#
##            ,nn.Sigmoid(), nn.Linear(H,1),nn.Sigmoid()) 
##    H2=round(H/2)
#    toy_model2=nn.Sequential(nn.Linear(L,H2))
##    ,nn.Sigmoid(), nn.Linear(H2,1),nn.Sigmoid()) 
#
#    ppsi=Psi(toy_model,toy_model2, L, form=Form)
#    
#    return ppsi

''' ########## Expand the O_omega routines to calculate grad of E ##########'''

# function to apply multipliers to the expectation value O_local expression 
def Exp_val(mat,wvf):
    
    if len(np.shape(mat))==0:
        O_l= np.sum(mat*np.abs(wvf.T)**2)\
        /(np.matmul(np.conjugate(wvf.T),wvf))
    else:
        O_l= np.sum(np.matmul((np.abs(wvf.T)**2),mat))\
        /(np.matmul(np.conjugate(wvf.T),wvf))
    
    return O_l

''' ######################## ANGLE COMPONENT CALC ##################### '''

H=2*L
ppsi_mod=psi_init(L,H,'euler') # without mult, initializes params randomly

## HAVE TO GENERATE THE SAMPLES VIA MONTE CARLO! Otherwise may not be wieghted correctly ##
sb=ppsi_mod.sample_MH(burn_in,spin=0.5)
s=torch.tensor(ppsi_mod.sample_MH(N_samples,spin=0.5, s0=sb[-1]),dtype=torch.float32)
modi_params=list(ppsi_mod.imag_comp.parameters())

[H_nn, H_b]=ppsi_mod.O_local(nn_interaction,s.numpy()),ppsi_mod.O_local(b_field,s.numpy())
E_loc=np.sum(H_nn+H_b,axis=1)

angle_net=copy.deepcopy(ppsi_mod)

#start = time.time()
#autograd_hacks.add_hooks(ppsi_mod.imag_comp)
outi=ppsi_mod.imag_comp(s)
#outi.mean().backward()
#autograd_hacks.compute_grad1(ppsi_mod.imag_comp)

# what we calculated the gradients should be
mult=torch.tensor(2*np.imag(-np.conj(E_loc)),dtype=torch.float)

ppsi_mod.imag_comp.zero_grad()

ppsi_mod.imag_comp.zero_grad()
ppsi_mod.energy_gradient(s,E_loc)
p_i=list(ppsi_mod.imag_comp.parameters())
module_g0=p_i[0].grad[0][0]
ppsi_mod.imag_comp.zero_grad()
ppsi_mod.energy_gradient1(s,E_loc)
p_i=list(ppsi_mod.imag_comp.parameters())
module_g1=p_i[0].grad[0][0]

#for param in modi_params:
#    if len(param.size())==2:
#        param.grad=torch.einsum("i,ijk->ijk",mult,param.grad1).mean(0)
#    elif len(param.size())==1:
#        param.grad=torch.einsum("i,ik->ik",mult,param.grad1).mean(0)

ppsi_mod.imag_comp.zero_grad()
(outi*mult[:,None]).mean().backward()
#end=time.time(); print(end-start)

pars=list(ppsi_mod.imag_comp.parameters())
grad0=pars[0].grad[0][0].item() 
modi_params=list(angle_net.imag_comp.parameters())

#[p1_r,p1_i]=ppsi_mod.apply_energy_gradient(s,E_loc,np.mean(E_loc),0.03)
#print(p1_i-grad0) # testing to make sure algorithm is working the same.

dw=0.001 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    modi_params[0][0][0]=modi_params[0][0][0]+dw

wvf0=ppsi_mod.complex_out(torch.tensor(s2,dtype=torch.float))
wvf1=angle_net.complex_out(torch.tensor(s2,dtype=torch.float))
E_tot0=np.matmul(np.matmul(np.conjugate(wvf0.T),H_tot),wvf0)/(np.matmul(np.conjugate(wvf0.T),wvf0))
E_tot1=np.matmul(np.matmul(np.conjugate(wvf1.T),H_tot),wvf1)/(np.matmul(np.conjugate(wvf1.T),wvf1))
dif=(E_tot1-E_tot0)/dw

print('\n\n #################### Angle ######################### \n',\
      'Pytorch derivative: ', grad0,'\n DE/dw with wavefunction: '\
  , dif, ' \n pytorch derivative relative error: ', \
  np.abs(grad0-dif)/dif, '\n Ratio: ', grad0/dif, \
  '\n method values: ',  module_g0.item(), module_g1.item(), '\n\n')

print('Exact energy: ', E_tot0, '\n vs sampled energy: ', np.mean(E_loc), \
      '\n with relative error: ', np.abs(np.mean(E_loc)-E_tot0)/E_tot0,'\n\n')

# Here calculate the base (unaltered) equivalent expression using O_local 
[H_nn_ex, H_b_ex]=ppsi_mod.O_local(nn_interaction,s2),ppsi_mod.O_local(b_field,s2)
E_loc=np.sum(H_nn_ex+H_b_ex,axis=1)
E0 = Exp_val(E_loc,wvf0)

# now, see if the derived alteration to the above expression accounts for the 
# observed diferential with dw
# get Ok on a per sample basis
Ok=np.zeros([np.shape(s2)[0],1],dtype=complex)
out=ppsi_mod.imag_comp(torch.tensor(s2,dtype=torch.float))
pars=list(ppsi_mod.imag_comp.parameters())
for ii in range(np.shape(s2)[0]):
    ppsi_mod.imag_comp.zero_grad()
    out[ii].backward(retain_graph=True) 

    Ok[ii]=1j*pars[0].grad[0][0].numpy()

Ok=np.squeeze(Ok)

# now for the full derivate expression using the original wavefunction
deriv_E0=Exp_val(np.conj(Ok)*E_loc,wvf0)+Exp_val(Ok*np.conj(E_loc),wvf0)-\
Exp_val(E_loc,wvf0)*(Exp_val(np.conj(Ok),wvf0)+Exp_val(Ok,wvf0))

# all equivalent methods
Force=Exp_val(np.conj(Ok)*E_loc,wvf0)-Exp_val(E_loc,wvf0)*Exp_val(np.conj(Ok),wvf0)
deriv_E01= Force+np.conj(Force)

deriv_E02=Exp_val(np.squeeze(2*np.real((np.conj(E_loc)-\
                    np.conj(Exp_val(E_loc,wvf0)))*Ok)),wvf0)

S=np.linalg.pinv(Exp_val(np.matmul(Ok,Ok.T),wvf0)-Exp_val(Ok,wvf0)*Exp_val(Ok.T,wvf0))

print('\n Expecation val deriv: ', deriv_E0, '\n vs numerical wvf energy diff: ', dif)

''' ######################## MODULUS COMPONENT CALC ##################### '''

ppsi_mod=psi_init(L,H,'euler') # without mult, initializes params randomly
mod_net=copy.deepcopy(ppsi_mod)
sb=ppsi_mod.sample_MH(burn_in,spin=0.5)
s=torch.tensor(ppsi_mod.sample_MH(N_samples,spin=0.5, s0=sb[-1]),dtype=torch.float32)

[H_nn, H_b]=ppsi_mod.O_local(nn_interaction,s.numpy()),ppsi_mod.O_local(b_field,s.numpy())
E_loc=np.sum(H_nn+H_b,axis=1)
E0=np.real(np.mean(E_loc))

outr=ppsi_mod.real_comp(s)

mult=torch.tensor((2*np.real(np.conj(E_loc)-np.conj(E0))))

# what we calculated the gradients should be
(outr.log()*mult[:,None]).mean().backward()

pars=list(ppsi_mod.real_comp.parameters())
grad0=pars[0].grad[0][0].item() 

#[p1_r,p1_i]=ppsi_mod.apply_energy_gradient(s,E_loc,np.mean(E_loc),0.03)
#print(p1_r-grad0) # testing to make sure algorithm is working the same.

pars2=list(mod_net.real_comp.parameters())
dw=0.001 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    pars2[0][0][0]=pars2[0][0][0]+dw

wvf0=ppsi_mod.complex_out(torch.tensor(s2,dtype=torch.float))
wvf1=mod_net.complex_out(torch.tensor(s2,dtype=torch.float))
E_tot0=np.matmul(np.matmul(np.conjugate(wvf0.T),H_tot),wvf0)/(np.matmul(np.conjugate(wvf0.T),wvf0))
E_tot1=np.matmul(np.matmul(np.conjugate(wvf1.T),H_tot),wvf1)/(np.matmul(np.conjugate(wvf1.T),wvf1))
dif=(E_tot1-E_tot0)/dw

ppsi_mod.real_comp.zero_grad()
ppsi_mod.energy_gradient(s,E_loc)
p_r=list(ppsi_mod.real_comp.parameters())
module_g0=p_r[0].grad[0][0]
ppsi_mod.real_comp.zero_grad()
ppsi_mod.energy_gradient1(s,E_loc)
p_r=list(ppsi_mod.real_comp.parameters())
module_g1=p_r[0].grad[0][0]

print('\n\n #################### Modulus ######################### \n',\
      'Pytorch derivative: ', grad0,'\n DE/dw with wavefunction: '\
  , dif, ' \n pytorch derivative relative error: ', \
  np.abs(grad0-dif)/dif, '\n Ratio: ', grad0/dif, \
  '\n method values: ',  module_g0.item(), module_g1.item(), '\n\n')

print('Exact energy: ', E_tot0, '\n vs sampled energy: ', np.mean(E_loc), 
      '\n with relative error: ', np.abs(np.mean(E_loc)-E_tot0)/E_tot0,'\n\n')

# Here calculate the base (unaltered) equivalent expression using O_local 
[H_nn_ex, H_b_ex]=ppsi_mod.O_local(nn_interaction,s2),ppsi_mod.O_local(b_field,s2)
E_loc=np.sum(H_nn_ex+H_b_ex,axis=1)
    
# get Ok on a per sample basis
Ok=np.zeros([np.shape(s2)[0],1],dtype=complex)
out=ppsi_mod.real_comp(torch.tensor(s2,dtype=torch.float))
pars=list(ppsi_mod.real_comp.parameters())
for ii in range(np.shape(s2)[0]):
    ppsi_mod.real_comp.zero_grad()
    out[ii].backward(retain_graph=True) 

    Ok[ii]=pars[0].grad[0][0].numpy()/out[ii].detach().numpy()
    
Ok=np.squeeze(Ok)

# now for the full derivate expression using the original wavefunction
deriv_E0=Exp_val(np.conj(Ok)*E_loc,wvf0)+Exp_val(Ok*np.conj(E_loc),wvf0)-\
Exp_val(E_loc,wvf0)*(Exp_val(np.conj(Ok),wvf0)+Exp_val(Ok,wvf0))

#Ok=grad_list[0].numpy()
#S=np.linalg.pinv(Exp_val(np.matmul(Ok,Ok.T),wvf0)-Exp_val(Ok,wvf0)*Exp_val(Ok.T,wvf0))

print('\n Expecation val deriv: ', deriv_E0, '\n vs numerical wvf energy diff: ', dif)

''' Now with vector version '''

''' ######################## REAL COMPONENT CALC ######################### '''

ppsi_vec=psi_init(L,H,'vector')  # without mult, initializes params randomly
real_net=copy.deepcopy(ppsi_vec)

sb=ppsi_vec.sample_MH(burn_in,spin=0.5)
s=torch.tensor(ppsi_vec.sample_MH(N_samples,spin=0.5, s0=sb[-1]),dtype=torch.float)

psi0=ppsi_vec.complex_out(s).squeeze()

[H_nn, H_b]=ppsi_vec.O_local(nn_interaction,s.numpy()),ppsi_vec.O_local(b_field,s.numpy())
E_loc=np.sum(H_nn+H_b,axis=1)
E0=np.real(np.mean(E_loc))

ppsi_vec.real_comp.zero_grad()
autograd_hacks.add_hooks(ppsi_vec.real_comp)
outr=ppsi_vec.real_comp(s)
outr.mean().backward()
autograd_hacks.compute_grad1(ppsi_vec.real_comp)
autograd_hacks.clear_backprops(ppsi_vec.real_comp)

m=1/psi0
mult=torch.tensor(np.real(2*(np.conj(E_loc)-np.conj(E0))/psi0),dtype=torch.float)

ppsi_vec.real_comp.zero_grad()

pars=list(ppsi_vec.real_comp.parameters())

for param in pars:
    if len(param.size())==2:
        param.grad=torch.einsum("i,ijk->ijk",mult,param.grad1).mean(0)
    elif len(param.size())==1:
        param.grad=torch.einsum("i,ik->ik",mult,param.grad1).mean(0)    
grad0=pars[0].grad[0][0].item()
print(grad0)

dw=0.001 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    pars[0][0][0]=pars[0][0][0]+dw

wvf0=real_net.complex_out(torch.tensor(s2,dtype=torch.float))
wvf1=ppsi_vec.complex_out(torch.tensor(s2,dtype=torch.float))
E_tot0=np.matmul(np.matmul(np.conjugate(wvf0.T),H_tot),wvf0)/(np.matmul(np.conjugate(wvf0.T),wvf0))
E_tot1=np.matmul(np.matmul(np.conjugate(wvf1.T),H_tot),wvf1)/(np.matmul(np.conjugate(wvf1.T),wvf1))
dif=(E_tot1-E_tot0)/dw

real_net.real_comp.zero_grad()
real_net.energy_gradient(s,E_loc,E0)
p_r=list(real_net.real_comp.parameters())
module_g0=p_r[0].grad[0][0]
real_net.real_comp.zero_grad()
real_net.energy_gradient1(s,E_loc,E0)
p_r=list(real_net.real_comp.parameters())
module_g1=p_r[0].grad[0][0]

print('\n\n #################### Real COMP ######################### \n',\
      'Pytorch derivative: ', grad0,'\n DE/dw with wavefunction: '\
  , dif, ' \n pytorch derivative relative error: ', \
  np.abs(grad0-dif)/dif, '\n Ratio: ', grad0/dif, \
  '\n method values: ',  module_g0.item(), module_g1.item(), '\n\n')

print('Exact energy: ', E_tot0, '\n vs sampled energy: ', np.mean(E_loc), 
      '\n with relative error: ', np.abs(np.mean(E_loc)-E_tot0)/E_tot0,'\n\n')

# Here calculate the base (unaltered) equivalent expression using O_local 
[H_nn_ex, H_b_ex]=real_net.O_local(nn_interaction,s2),real_net.O_local(b_field,s2)
E_loc=np.sum(H_nn_ex+H_b_ex,axis=1)
    
psi=real_net.complex_out(torch.tensor(s2,dtype=torch.float))
Ok=np.zeros([np.shape(s2)[0],1],dtype=complex)
out=real_net.real_comp(torch.tensor(s2,dtype=torch.float))
pars=list(real_net.real_comp.parameters())
for ii in range(np.shape(s2)[0]):
    real_net.real_comp.zero_grad()
    out[ii].backward(retain_graph=True) 

    Ok[ii]=pars[0].grad[0][0].numpy()/psi[ii]
    
Ok=np.squeeze(Ok)

deriv_E0=Exp_val(np.conj(Ok)*E_loc,wvf0)+Exp_val(Ok*np.conj(E_loc),wvf0)-\
Exp_val(E_loc,wvf0)*(Exp_val(np.conj(Ok),wvf0)+Exp_val(Ok,wvf0))

print('\n Expecation val deriv: ', deriv_E0, '\n vs numerical wvf energy diff: ', dif)

''' ######################## IMAGINARY COMPONENT CALC ##################### '''
# finally for complex vec i
ppsi_vec=psi_init(L,H,'vector') # without mult, initializes params randomly
imag_net=copy.deepcopy(ppsi_vec)

sb=ppsi_vec.sample_MH(burn_in,spin=0.5)
s=torch.tensor(ppsi_vec.sample_MH(N_samples,spin=0.5, s0=sb[-1]),dtype=torch.float)

psi0=ppsi_vec.complex_out(s).squeeze() # the original psi

[H_nn, H_b]=ppsi_vec.O_local(nn_interaction,s.numpy()),ppsi_vec.O_local(b_field,s.numpy())
E_loc=np.sum(H_nn+H_b,axis=1)
E0=np.mean(E_loc) 

imag_net.imag_comp.zero_grad()
imag_net.energy_gradient(s,E_loc)
p_r=list(imag_net.imag_comp.parameters())
module_g0=p_r[0].grad[0][0]
imag_net.imag_comp.zero_grad()
imag_net.energy_gradient1(s,E_loc)
p_r=list(imag_net.imag_comp.parameters())
module_g1=p_r[0].grad[0][0]

autograd_hacks.add_hooks(ppsi_vec.imag_comp)
outi=ppsi_vec.imag_comp(s)
outi.mean().backward()
autograd_hacks.compute_grad1(ppsi_vec.imag_comp)
autograd_hacks.clear_backprops(ppsi_vec.imag_comp)

mult=torch.tensor(np.real(2j*(np.conj(E_loc)-np.conj(E0))/psi0),dtype=torch.float)

ppsi_vec.imag_comp.zero_grad()

pars=list(ppsi_vec.imag_comp.parameters())

for param in pars:
    if len(param.size())==2:
        param.grad=torch.einsum("i,ijk->ijk",mult,param.grad1).mean(0)
    elif len(param.size())==1:
        param.grad=torch.einsum("i,ik->ik",mult,param.grad1).mean(0)    

grad0=pars[0].grad[0][0].item()

dw=0.001 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    pars[0][0][0]=pars[0][0][0]+dw

wvf0=imag_net.complex_out(torch.tensor(s2,dtype=torch.float))
wvf1=ppsi_vec.complex_out(torch.tensor(s2,dtype=torch.float))
E_tot0=np.matmul(np.matmul(np.conjugate(wvf0.T),H_tot),wvf0)/(np.matmul(np.conjugate(wvf0.T),wvf0))
E_tot1=np.matmul(np.matmul(np.conjugate(wvf1.T),H_tot),wvf1)/(np.matmul(np.conjugate(wvf1.T),wvf1))
dif=(E_tot1-E_tot0)/dw

print('\n\n #################### IMAG COMP ######################### \n',\
      'Pytorch derivative: ', grad0,'\n DE/dw with wavefunction: '\
  , dif, ' \n pytorch derivative relative error: ', \
  np.abs(grad0-dif)/dif, '\n Ratio: ', grad0/dif, \
  '\n method values: ',  module_g0.item(), module_g1.item(), '\n\n')

print('Exact energy: ', E_tot0, '\n vs sampled energy: ', np.mean(E_loc), 
      '\n with relative error: ', np.abs(np.mean(E_loc)-E_tot0)/E_tot0,'\n\n')

# Here calculate the base (unaltered) equivalent expression using O_local 
[H_nn_ex, H_b_ex]=imag_net.O_local(nn_interaction,s2),imag_net.O_local(b_field,s2)
E_loc=np.sum(H_nn_ex+H_b_ex,axis=1)
    
psi=imag_net.complex_out(torch.tensor(s2,dtype=torch.float))
Ok=np.zeros([np.shape(s2)[0],1],dtype=complex)
out=imag_net.imag_comp(torch.tensor(s2,dtype=torch.float))
pars=list(imag_net.imag_comp.parameters())
for ii in range(np.shape(s2)[0]):
    imag_net.imag_comp.zero_grad()
    out[ii].backward(retain_graph=True) 

    Ok[ii]=1j*pars[0].grad[0][0].numpy()/psi[ii]
    
Ok=np.squeeze(Ok)

deriv_E0=Exp_val(np.conj(Ok)*E_loc,wvf0)+Exp_val(Ok*np.conj(E_loc),wvf0)-\
Exp_val(E_loc,wvf0)*(Exp_val(np.conj(Ok),wvf0)+Exp_val(Ok,wvf0))

print('\n Expecation val deriv: ', deriv_E0, '\n vs numerical wvf energy diff: ', dif)