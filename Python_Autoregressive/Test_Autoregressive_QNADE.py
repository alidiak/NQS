#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:15:18 2020

Testing the Quantum Neural Autoregressive Density Estimator (QNADE):
Sections 1-2: are model and function definition, respectively
Section 3: tests sampling routine and sampling convergence to |Psi|^2
Section 4: tests energy calculation and convergence with increasing samples
Sections 5-7: tests accuracy and time cost of energy gradient
Section 8: tests optimization routine

@author: Alex Lidiak
"""

import numpy as np
from autograd_hacks_master import autograd_hacks
import matplotlib.pyplot as plt
import torch
from NQS_pytorch import Psi, Op, kron_matrix_gen
import time
import itertools
import copy
import torch.nn as nn

# Load libraries and the models therein
import sys
sys.path.insert(1, '/home/alex/Documents/QML_Research/Variational_Learning_'\
  'Implementations/Python_Autoregressive/models')
#from qnade import QNADE
#from qnade_custom import QNADE_custom

# Overarching torch datatype and precision
datatype=torch.double

# system parameters
b= 1 # b-field strength
J= -1 # nearest neighbor interaction strength, J>0 ferromagnetic, J<0 anti-ferro
L = 3 # system size
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
op_list=[] 
for ii in range(2): # make sure to change to match Hamiltonian entered above
    op_list.append(sigmaz)
H_szsz=-J*kron_matrix_gen(op_list,len(evals),L,'periodic').toarray()

op_list2=[]
op_list2.append(sigmax)
H_sx=b*kron_matrix_gen(op_list2,len(evals),L,'periodic').toarray()

H_tot=H_szsz+H_sx
min_E=np.min(np.linalg.eigvals(H_tot))

s2=np.array(list(itertools.product(evals,repeat=L)))

''' ## Method Definition (modified/testing version) ## '''

def QNADE_pass2(model, N_samples=None, x=None, requires_grad=False, E_arg=None): 
            
    if N_samples is None and x is None: 
        raise ValueError('Must enter spin states for Psi calculation or the number of samples to be generated')
    if N_samples is None and x is not None: N_samples, sample = x.shape[0], False
    if N_samples is not None and x is None: 
        sample = True
        x = torch.zeros([N_samples,model.L],dtype=model.dtype)  # make a stand in x
    
    # the full Psi is a product of the conditionals, making a running product easy
    PPSI=np.ones([N_samples],dtype=np.complex128) # if multiplying
    #PPSI=np.zeros([N_samples],dtype=np.complex128)  # if adding logs
    
    # number of outputs we must get for the output layer
    nevals = len(model.evals)
    
    if requires_grad:
        
        pars=list(model.real_comp.parameters())
        gradii = [[] for i in range(len(pars))] # create and assign memory for the gradient
        if E_arg is None: O_k, E_grad = [[] for i in range(len(pars))], 0
        else: E_grad, O_k= [[] for i in range(len(pars))], 0
        for rr in range(len(pars)):
            if len(pars[rr].size())==2:
                [sz1,sz2]=[pars[rr].size(0),pars[rr].size(1)]
            else:
                [sz1,sz2]=[pars[rr].size(0),1]
            if E_arg is None: O_k[rr]=np.zeros([N_samples, sz1,sz2, 2],dtype=complex)
            else: E_grad[rr]=np.zeros([sz1,sz2, 2],dtype=complex)
            gradii[rr]=np.zeros([N_samples, sz1,sz2,len(evals), 2])
                
    for d in range(model.L):
        if requires_grad:
            if not hasattr(model.real_comp,'autograd_hacks_hooks'):             
                autograd_hacks.add_hooks(model.real_comp)
            if not hasattr(model.imag_comp,'autograd_hacks_hooks'):             
                autograd_hacks.add_hooks(model.imag_comp)
            
        # masks enforce the autoregressive property
        mask=torch.zeros_like(x)
        mask[:,0:(d)]=1
        # TODO: use maskes to enforce an arbitrary order to the inputs
        # this will also involve redefining Psi to an average over the order
        # implementations.
        
        outr=model.real_comp(mask*x)
        outi=model.imag_comp(mask*x)
    
        vi_dr=outr[:,nevals*d:(nevals*(d+1))]
        vi_di=outi[:,nevals*d:(nevals*(d+1))]
        
        # The Quantum-NADE deviates from a NADE in having a real and imag comp
        # Here we can use both vi to generate a complex vi that is the 
        # basis of our calculations and sampling
        # TODO add form options other than exponential
        vi = np.exp(vi_dr.detach().numpy()+1j*vi_di.detach().numpy())
        
        # Normalization and formation of the conditional psi
        exp_vi=np.exp(vi) # unnorm prob of evals 
        norm_const=np.sqrt(np.sum(np.power(np.abs(exp_vi),2),1))
        psi=np.einsum('ij,i->ij', exp_vi, 1/norm_const) 
        
        # Sampling probability is determined by the born rule in QM
        if sample:
            born_psi=np.power(np.abs(psi),2)
            assert np.all(np.sum(born_psi,1)-1<1e-6), "Psi not normalized correctly"
            
            # sampling routine:
            probs = born_psi.copy()
            for ii in range(1, probs.shape[1]): # accumulate prob ranges for easy 
                probs[:,ii] = probs[:,ii]+probs[:,ii-1] # sampling with 0<alpha<1
            
            a=np.random.rand(N_samples)[:,None]
            samplepos=np.sum(probs<a,1) # find the sample position in eval list
            
            # corrected error where sometimes a too large index occurs here
            # This is a rough fix... But not sure what a better way to ensure it would be
            if np.any(samplepos==len(model.evals)):
                samplepos[samplepos==len(model.evals)] -= 1 
            
            x[:,d] = torch.tensor(model.evals[samplepos], dtype=model.dtype) # sample
            # End sampling routine
        
        else:
            xd = x[:,d:d+1]
            if d==0: samples=xd # just checking the iterations
            else: samples = torch.cat((samples,xd),dim=1) 
            # find the s_i for psi(s_i), which is to be accumulated for PPSI
            samplepos = (xd==model.evals[1]).int().numpy().squeeze()
            # TODO this definitely won't work for non-binary evals, need to
            # extend functionality to any set of evals. Also careful about evals[0] vs [1]
            
        if requires_grad:
            exp_t=np.exp(2*np.real(vi)) # useful terms for the O_k calculation
            norm_term=np.sum(exp_t,1)
            
            for cc in range(2): # loop for each component: real and imag
                if cc==0:
                    psi_i=outr[:,nevals*d:(nevals*(d+1))]
                    use_comp= model.real_comp
                else: 
                    psi_i=outi[:,nevals*d:(nevals*(d+1))]
                    use_comp=model.imag_comp
                
                pars=list(use_comp.parameters())
                for jj in range(len(model.evals)):
                    # the eval specific gradient
                    psi_i[:,jj].mean(0).backward(retain_graph=True)
                    
                    autograd_hacks.compute_grad1(use_comp)
                    autograd_hacks.clear_backprops(use_comp) 
                    for rr in range(len(pars)):
                        if len(pars[rr].size())==1:
                            gradii[rr][...,jj, cc]=pars[rr].grad1.numpy()[...,None]
                        else:
                            gradii[rr][...,jj, cc]=pars[rr].grad1.numpy()
                    
                for rr in range(len(pars)): # have to include all pars 
                    grad=gradii[rr][...,cc]
                    
                    # derivative term (will differ depending on ansatz 'form')
                    if model.form.lower()=='exponential':
                        if cc==0: dvi = np.einsum('il,ijkl->ijkl', vi, grad)
                        else: dvi =  np.einsum('il,ijkl->ijkl', 1j*vi, grad)
            
                    st_mult =  np.sum(np.einsum('il,ijkl->ijkl', exp_t, np.real(dvi)),-1)
                    sec_term=np.einsum('i,ijk->ijk', 1/norm_term, st_mult)
                        
                    if E_arg is None:              
                        O_k[rr][...,cc] += np.real(dvi[range(N_samples),...,samplepos]-sec_term)
                    else: 
                        E_grad[rr][...,cc] += np.mean(2*np.real(np.einsum('i,ijk->ijk', E_arg, \
                              (dvi[range(N_samples),...,samplepos]-sec_term))),0)
                            
        else: E_grad, O_k= 0, 0

        # Multiplicitavely accumulate PPSI based on which sample (s) was sampled
        PPSI=PPSI*psi[range(N_samples),samplepos]
        
        # PPSI may only make sense when inputing an x to get the wvf for...
    
    model.wvf=PPSI
    
    return PPSI, x, E_grad, O_k

'''################ Test Sample Distribution vs |Psi(s)|^2 #################'''

def direct_sampling(wvf, N_samples):
    
    samplepos=np.zeros([N_samples,1]) # record the sampled states
    probs=np.power(np.abs(wvf),2)
    
    for ii in range(1,len(probs)):
        probs[ii]=probs[ii]+probs[ii-1] # accumulate prob ranges for easy 
                                        # sampling with 0<alpha<1
    
    for jj in range(N_samples):
        a=np.random.rand()
        samplepos[jj]=np.sum(probs<a)
    
    return samplepos

#ppsi=psi_init(L,hidden_layer_sizes,nout,'exponential',datatype)
H=2*L
#autoreg_real_net=nn.Sequential(nn.Linear(L,H),nn.Sigmoid(), nn.Linear(H,len(evals)*L)) # for autoregressive final layer must be nevals*L
#autoreg_imag_net=nn.Sequential(nn.Linear(L,H), nn.Sigmoid(), nn.Linear(H,len(evals)*L)) 
autoreg_real_net=nn.Sequential(nn.Linear(L,H), nn.Sigmoid(), nn.Linear(H,H), \
    nn.Sigmoid(), nn.Linear(H,len(evals)*L)) # for autoregressive final layer output must be nevals*L
autoreg_imag_net=nn.Sequential(nn.Linear(L,H), nn.Sigmoid(), nn.Linear(H,H), \
    nn.Sigmoid(), nn.Linear(H,len(evals)*L)) 

ppsi = Psi(autoreg_real_net, autoreg_imag_net, L, form='exponential', dtype=datatype, autoregressive=True)

# Alternative definitions (worked way up to final ppsi.QNADE implementation)
#ppsi = QNADE(L, L, [-1,1])
#ppsi = QNADE_custom(autoreg_real_net, autoreg_imag_net, evals)

#wvf, _ = ppsi(x=torch.tensor(s2,dtype=datatype)) # Syntax if QNADE or QNADE_custom used
wvf,_,_,_ = QNADE_pass2(ppsi, x=torch.tensor(s2,dtype=datatype))

plt.figure()
plt.bar(range(0,len(s2)), abs(wvf)**2)
plt.title('Probability distribution, |Psi(s)|^2')

#N_samp_list = np.logspace(4,7,10)
N_resamples=1 # doesn't really seem to help, default should be 1
N_samp_list=[10000]
avg_rel_err = np.zeros([len(N_samp_list),1])
direct_sampling_err = np.zeros([len(N_samp_list),1])
for kk in range(len(N_samp_list)):
    N_samples=int(round(N_samp_list[kk]))
    
    alt_wvf, s, _, _ = QNADE_pass2(ppsi, N_samples=N_samples)
    s=s.numpy()

    # evaluating direct sampling as comparison
    samplepos = direct_sampling(wvf,N_samples) 

    h=[]; ind=[]; h_direct=[]
    for ii in range(0,len(s2)):
        h.append(np.sum((np.sum(s==s2[ii],1)==L)))
        ind.append(np.where(np.sum(s==s2[ii],1)==L))
        h_direct.append(np.sum(samplepos==ii))

    rel_err= np.abs(((abs(wvf)**2)-h/np.sum(h))/(abs(wvf)**2))
    ds_err=np.abs(((abs(wvf)**2)-h_direct/np.sum(h_direct))/(abs(wvf)**2))
    
    direct_sampling_err[kk]=np.mean(ds_err)
    avg_rel_err[kk]=np.mean(rel_err)

    if len(N_samp_list)==1:
        plt.figure()
        plt.bar(range(0,len(s2)), h/np.sum(h))
        plt.title('Histogram of Samples, autoregressive sampling with ' + str(N_samples) + ' samples' )

        plt.figure()
        plt.bar(range(0,len(s2)), h_direct/np.sum(h_direct))
        plt.title('Histogram of Samples, direct sampling with ' + str(N_samples)+ ' samples')
        
        plt.figure()
        plt.plot(rel_err,'o-')
        plt.title('relative error of sample distribution for # samples=' + str(N_samples) )

plt.figure()
plt.plot(np.log10(N_samp_list),avg_rel_err,'o-')
plt.plot(np.log10(N_samp_list),direct_sampling_err,'x-')
plt.plot(np.log10(N_samp_list), 1/np.sqrt(N_samp_list),'.-')
plt.xlabel('Log10(N_samples)'), plt.ylabel('Average Relative Error')
plt.legend(('Autoregressive Relative Error','Direct Sampling Error', '1/sqrt(N_samples)'))

'''################## Test Energy Calculation #########################'''

wvf, _,_,_ = QNADE_pass2(ppsi, x=torch.tensor(s2,dtype=datatype))

if len(wvf.shape)==1: wvf=wvf[:,None]
E_sx=np.matmul(np.matmul(np.conjugate(wvf.T),H_sx),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))
E_szsz=np.matmul(np.matmul(np.conjugate(wvf.T),H_szsz),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))
E_tot=np.matmul(np.matmul(np.conjugate(wvf.T),H_tot),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))

#N_samp_list= np.logspace(4,7,10)
N_samp_list = [1e+5]
E_rel_err=np.zeros([len(N_samp_list),1])
for kk in range(len(N_samp_list)):
    N_samples=int(round(N_samp_list[kk]))
    alt_wvf, s ,_,_ = QNADE_pass2(ppsi, N_samples)

    H_nn=ppsi.O_local(nn_interaction,s.numpy())
    H_b=ppsi.O_local(b_field,s.numpy())

    print('For psi= \n', wvf, '\n\n the energy (using exact H) is: ', E_tot, '\n while that ' \
          'predicted with the O_local function is: ', np.sum(np.mean(H_b+H_nn,axis=0)), \
          '\n\n for the exact Sx H: ', E_sx, ' vs ',np.sum(np.mean(H_b,axis=0)), \
          '\n\n for exact SzSz H: ', E_szsz ,' vs ', np.sum(np.mean(H_nn,axis=0)))
    
    E_rel_err[kk]= np.abs((E_tot-np.sum(np.mean(H_b+H_nn,axis=0))))/np.abs(E_tot)

plt.figure()
plt.plot(np.log10(N_samp_list), E_rel_err, 'o-'); 
plt.xlabel('log10(Sample Size)'); plt.ylabel('Energy Relative Error')
plt.title('Autoregressive Sampled Energy Estimate vs. Number of Samples')

''' Ensuring that <psi|H|psi> = \sum_s |psi(s)|^2 e_loc(s)   '''

H_szsz_ex=ppsi.O_local(nn_interaction,s2)
H_sz_ex=ppsi.O_local(b_field,s2)
O_loc_analytic= np.sum(np.matmul((np.abs(wvf.T)**2),(H_szsz_ex+H_sz_ex)))\
 /(np.matmul(np.conjugate(wvf.T),wvf))
E_exact=np.matmul(np.matmul(np.conjugate(wvf.T),H_tot),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))

print('\n\n Energy using O_local in the analytical expression: ',O_loc_analytic, \
      '\n vs. that calculated with matrices: ', E_exact )

'''######## Potentially helpful functions for O_k and grad computation #######'''

# function to apply multipliers to the expectation value O_local expression 
def Exp_val(mat,wvf):
    if len(np.shape(mat))==0:
        O_l= np.sum(mat*np.abs(wvf.T)**2)\
        /(np.matmul(np.conjugate(wvf.T),wvf))
    else:
        O_l= np.sum(np.matmul((np.abs(wvf.T)**2),mat))\
        /(np.matmul(np.conjugate(wvf.T),wvf))
    return O_l

# Little algorithm to test the above Ok construction. Will compare the above 
# fully computed Ok for any legitimate combination of parameter indices (i.e.  
# unmasked or existant (bias layers will not have a third parameter index)). 
def Ok_test(model, O_k,s, dw=0.001, outprint=False):
    
    for cc in range(2):
        if cc==0: comp, pars = 'real', list(model.real_comp.parameters())
        else: comp, pars= 'imag',list(model.imag_comp.parameters())
        
        rel_err_list=[]
        for rr in range(len(pars)):
            for ss in range(pars[rr].size(0)):
                if len(pars[rr].size())==2:
                    for tt in range(pars[rr].size(1)):
                        rel_err_list.append(Ok_compare(model, comp, s, rr,ss,tt, O_k, dw, outprint))
                else:
                    rel_err_list.append(Ok_compare(model, comp, s, rr,ss,0,O_k, dw, outprint))
                    
        rel_err_list=np.array(rel_err_list)
        if not np.all( rel_err_list<=0.1 ):
            print('Not all Ok are calculated accurately to within a 10% error!')            
        print('Maximum relative error for ', comp,' component: ', np.max(rel_err_list) )
        
    return
                
def Ok_compare(model, comp, s, par_ind1,par_ind2,par_ind3, O_k, dw, outprint):
    ppsi=copy.deepcopy(model) # just so it's an equivalent starting spot each time
    if comp=='real': pars1, c_ind= list(ppsi.real_comp.parameters()), 0
    else: pars1, c_ind = list(ppsi.imag_comp.parameters()), 1
    
#    if s.dtype==torch.float32 or s.dtype==torch.double: s=s.numpy()
    
    with torch.no_grad():
        if len(pars1[par_ind1].shape)>1:
            pars1[par_ind1][par_ind2][par_ind3]=pars1[par_ind1][par_ind2][par_ind3]+dw
        else:
            pars1[par_ind1][par_ind2]=pars1[par_ind1][par_ind2]+dw
    
    wvf0,_,_,_ = QNADE_pass2(original_net,x=torch.tensor(s,dtype=datatype))
    wvf1,_,_,_ = QNADE_pass2(ppsi, x=torch.tensor(s,dtype=datatype))
    E_tot0=np.matmul(np.matmul(np.conjugate(wvf0.T),H_tot),wvf0)/(np.matmul(np.conjugate(wvf0.T),wvf0))
    E_tot1=np.matmul(np.matmul(np.conjugate(wvf1.T),H_tot),wvf1)/(np.matmul(np.conjugate(wvf1.T),wvf1))
    dif=(E_tot1-E_tot0)/dw
    
    [H_nn_ex, H_b_ex]=original_net.O_local(nn_interaction,s),original_net.O_local(b_field,s)
    E_loc=np.sum(H_nn_ex+H_b_ex,axis=1)
    
    if O_k[0].shape[0]==s.shape[0]: # means Ok=dln(Psi(s))/dw must be entered
        this_Ok=O_k[par_ind1][:,par_ind2,par_ind3, c_ind]
        
        deriv_E0=Exp_val(np.conj(this_Ok)*E_loc,wvf0)+Exp_val(this_Ok*np.conj(E_loc),wvf0)-\
        Exp_val(E_loc,wvf0)*(Exp_val(np.conj(this_Ok),wvf0)+Exp_val(this_Ok,wvf0))
        
    else: deriv_E0 = O_k[par_ind1][par_ind2,par_ind3, c_ind] # expect O_k to be the Energy gradient in this case
       
    # fixes error where nan is returned if both are 0.
    if np.abs(deriv_E0)==0 and np.abs(dif)==0: rel_err=0
    else: rel_err=np.abs((dif-deriv_E0)/deriv_E0)
    
    if outprint:
        print('\n Expecation (exact) deriv: ', deriv_E0,'\n vs numerical', \
        ' (Pytorch Gen) wvf energy diff: ', dif, '\n relative error: ', rel_err )
    
    if rel_err>=0.1:
        print('\n High relative error at parameter index: ', par_ind1, ' ', \
              par_ind2, ' ', par_ind3, '\n With numerical deriv: ', dif, \
              ' and pytorch estimated deriv: ', deriv_E0)        
    
    return rel_err

'''################## Test O_k Generation ###################'''

ppsi = Psi(autoreg_real_net, autoreg_imag_net, L, form='exponential', dtype=datatype, autoregressive=True)
original_net=copy.deepcopy(ppsi)

_,s,_,O_k=QNADE_pass2(ppsi,x=torch.tensor(s2,dtype=datatype), requires_grad=True)

# test if Ok is calculated accurately
Ok_test(original_net, O_k,s2, dw=0.01, outprint=False) #outprint option is to print out all errors

'''##### Test speed difference of QNADE with and without grad function #####'''
N_samples=int(1e+5)

start=time.time()
_, s, _,_ = QNADE_pass2(ppsi,N_samples, requires_grad=False)
end=time.time(); print('QNADE without grad:', end-start)

start=time.time()
_,_,_,O_k=QNADE_pass2(ppsi,x=s, requires_grad=True)
end=time.time(); print('QNADE with O_k:', end-start)

start=time.time()
_,_,_,O_k=QNADE_pass2(ppsi,N_samples, requires_grad=True)
end=time.time(); print('QNADE with O_k and Sampling:', end-start)

[H_nn_ex, H_b_ex]=original_net.O_local(nn_interaction,s.numpy()),original_net.O_local(b_field,s.numpy())
E_loc=np.sum(H_nn_ex+H_b_ex,axis=1)
E_argument=(np.conj(E_loc)-np.conj(np.mean(E_loc)))

start=time.time()
_,_,E_grad,_=QNADE_pass2(ppsi, x=s, requires_grad=True, E_arg=E_argument)
end=time.time(); print('QNADE with E_grad:', end-start)

# Doesn't really make sense to do this one as E_arg needs to be calculated during
# sample generation. #TODO: Add running E_arg during sample generation to inc. speed.
start=time.time()
_,_,_,E_grad=QNADE_pass2(ppsi, N_samples, requires_grad=True, E_arg=E_argument)
end=time.time(); print('QNADE with E_grad and Sampling:', end-start)

''' ## Finally Test Energy Gradient Calculation with Sampling ## '''
# E_grad over s2, may not be a fair test as there are only 8 samples and the underlying
# probability density distribution is not reflected in the s2. Can be very far off 
# The testing of the E_grad should be done over Autoregressively generated samples
N_samples=int(1e+6)

_, s, _,_ = QNADE_pass2(ppsi, N_samples, requires_grad=False)

[H_nn_ex, H_b_ex]=original_net.O_local(nn_interaction,s.numpy()),original_net.O_local(b_field,s.numpy())
E_loc=np.sum(H_nn_ex+H_b_ex,axis=1)
E_argument=(np.conj(E_loc)-np.conj(np.mean(E_loc)))

_,_,E_grad,_=QNADE_pass2(ppsi, x=s, requires_grad=True, E_arg=E_argument)

Ok_test(original_net, E_grad, s2, dw=0.001, outprint=False)
# WORKS!!!

'''##### Finally, optimize by combining sampling and gradient descent  #####'''
H=2*L
autoreg_real_net=nn.Sequential(nn.Linear(L,H), nn.Sigmoid(), nn.Linear(H,H), \
    nn.Sigmoid(), nn.Linear(H,len(evals)*L)) # for autoregressive final layer output must be nevals*L
autoreg_imag_net=nn.Sequential(nn.Linear(L,H), nn.Sigmoid(), nn.Linear(H,H), \
    nn.Sigmoid(), nn.Linear(H,len(evals)*L)) 
ppsi = Psi(autoreg_real_net, autoreg_imag_net, L, form='exponential', dtype=datatype, autoregressive=True)

# Enter simulation hyper parameters
N_iter=100
N_samples=int(1e+6)
lr=0.03
real_time_plot=True
exact_energy=False # TODO: isn't working with the autoregressive model for some reason

if real_time_plot:
    plt.figure()
    plt.axis([0, N_iter, min_E-0.5, L])
    plt.axhline(y=min_E,color='r',linestyle='-')

energy_n=np.zeros([N_iter,1])
for n in range(N_iter):
    
    if exact_energy and L<=14: # if want to test the energy without sampling
        s=torch.tensor(s2,dtype=datatype)
        wvf,_,_,_ = QNADE_pass2(ppsi,x=s)
        
        E_tot=np.matmul(np.matmul(np.conjugate(wvf.T),H_tot),wvf)\
        /(np.matmul(np.conjugate(wvf.T),wvf))
        
        # Need sampling, as s2 will have low prob states of Psi disproportionately represented
        # Get the energy at each iteration
        [H_nn, H_b]=ppsi.O_local(nn_interaction,s.numpy()),ppsi.O_local(b_field,s.numpy())
        energy_per_sample = np.sum(H_nn+H_b,axis=1)
        energy_n[n]=E_tot
    else:
        _, s, _,_ = QNADE_pass2(ppsi, N_samples)
        # Get the energy at each iteration
        [H_nn, H_b]=ppsi.O_local(nn_interaction,s.numpy()),ppsi.O_local(b_field,s.numpy())
        energy_per_sample=np.sum(H_nn+H_b,axis=1)
        energy_n[n] = np.real(np.mean(energy_per_sample))
    
    # calculate the energy gradient, updates pars in Psi object
    E_argument=(np.conj(energy_per_sample)-np.conj(np.mean(energy_per_sample)))
    _,_,E_grad,_=QNADE_pass2(ppsi, x=s, requires_grad=True, E_arg=E_argument)
    
    for cc in range(2):
        if cc==0: pars=list(ppsi.real_comp.parameters())
        else: pars=list(ppsi.imag_comp.parameters())
        for rr in range(len(pars)):
            pars[rr].grad=torch.tensor(np.real(E_grad[rr][...,cc])\
                ,dtype=datatype).squeeze()
    
#    lr=lr*0.99 # optional operation, reduces lr in simple iterative way
    ppsi.apply_grad(lr) # releases/updates parameters based on grad method (stored in pars.grad)

    if n%10==0:
        print('percentage of iterations complete: ', (n/N_iter)*100)
    
    if real_time_plot:
        if n>=1:
            plt.plot([n-1,n],[energy_n[n-1],energy_n[n]],'b-')
            plt.pause(0.05)
            plt.draw()













'''################## Test D v_i(s_i)/DW Calculation #########################'''

ppsi = Psi(autoreg_real_net, autoreg_imag_net, L, form='exponential', dtype=datatype, autoregressive=True)
original_net=copy.deepcopy(ppsi)

ppsi.real_comp.zero_grad()

pars1=list(ppsi.real_comp.parameters())

dw=0.001 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    pars1[0][0][0]=pars1[0][0][0]+dw

# Choose a specific s
s=torch.tensor(np.random.choice(evals,[1,L]),dtype=datatype)

# First test the autodifferentiation:
if not hasattr(original_net.real_comp,'autograd_hacks_hooks'):             
    autograd_hacks.add_hooks(original_net.real_comp)
out_0=original_net.real_comp(s)
autograd_hacks.clear_backprops(original_net.real_comp)
#outc=original_net.complex_out(s)
#vi=torch.tensor(np.real(outc),dtype=torch.float)
#(out_0*vi).mean().backward() # trying to multiply by vi then backprop 
                            #(have to make vi real, but complex values still 
                            # important for dE and ln(Psi))
grad0=torch.zeros([len(evals)*L,1])
for ii in range(len(evals)*L): # have to get the dpsi separately FROM EACH OUTPUT
    (out_0[0,ii]).backward(retain_graph=True)
    autograd_hacks.compute_grad1(original_net.real_comp)
    autograd_hacks.clear_backprops(original_net.real_comp)
    pars=list(original_net.real_comp.parameters())
    grad0[ii]=pars[0].grad1[0,0,0]

# Compare to dpsi
out_1=ppsi.real_comp(s)
dpsi=((out_1-out_0)/dw).detach().numpy() # just the real network output diff
print('Numerical dpsi_ij/dw: \n', dpsi, 'Pytorch dpsi_ij/dw: \n', grad0.T )
# WORKS! 

# Now calculate the numerical dvi and compare to analytical expression
outc0=original_net.complex_out(s)
outc1=ppsi.complex_out(s)
vi_dif=(outc1-outc0)/dw # test the difference between vis

test=dpsi*outc0 # dvi/dw (vi_dif) should equal dpsi/dw*vi (for exponential form)
print('\n numerical vi difference: \n', vi_dif, '\n vi*dpsi: \n', test)
# Works! Proof of concept for what the derivative should be for a given form

         
