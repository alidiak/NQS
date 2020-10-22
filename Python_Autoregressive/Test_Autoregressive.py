#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:15:18 2020

@author: alex
"""

import numpy as np
from autograd_hacks_master import autograd_hacks
import matplotlib.pyplot as plt
import torch
from NQS_pytorch import Psi, Op, kron_matrix_gen
import time
import itertools
import copy

# Load libraries and the models therein
import sys
sys.path.insert(1, '/home/alex/Documents/QML_Research/Variational_Learning_'\
  'Implementations/Python_Autoregressive/models')

# Overarching torch datatype and precision
datatype=torch.double

# system parameters
b=0.0   # b-field strength
J=1     # nearest neighbor interaction strength
L = 3   # system size
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

'''####### Define Neural Networks and initialization funcs for psi ########'''

# Neural Autoregressive Density Estimators (NADEs) output a list of 
# probabilities equal in size to the input. Futhermore, a softmax function is 
# handy as it ensures the output is probability like aka falls between 0 and 1.
# For a simple spin 1/2, only a single Lx1 output is needed as p(1)=1-P(-1),
# but with increasing number of eigenvalues, the probability and output becomes
# more complex. 

nout=len(evals)*L
hidden_layer_sizes=2*L

# NADE model in Pytorch Generative library by Eugen Hotaj
#import pytorch_generative as pg
#
#model_r=pg.models.NADE(L,hidden_layer_sizes)
#model_i=pg.models.NADE(L,hidden_layer_sizes)
#
#def psi_init(L, hidden_layer_sizes, nout=L, Form='exponential', dtype=torch.float):
#   
#    model_r=pg.models.NADE(L,hidden_layer_sizes)
#    model_i=pg.models.NADE(L,hidden_layer_sizes)
#
#    ppsi=Psi(model_r,model_i, L, form=Form,dtype=datatype, autoregressive=True)
#    
#    return ppsi

# NADE model by simonjisu
#from NADE_pytorch_master.model import NADE
#
#model_r=NADE(L,hidden_layer_sizes)
#model_i=NADE(L,hidden_layer_sizes)
#
#def psi_init(L, hidden_layer_sizes, nout=L, Form='exponential', dtype=torch.float):
#   
#    model_r=NADE(L,hidden_layer_sizes)
#    model_i=NADE(L,hidden_layer_sizes)
#
#    ppsi=Psi(model_r,model_i, L, form=Form,dtype=datatype, autoregressive=True)
#    
#    return ppsi

#The MADE coded by Andrej Karpath uses Masks to ensure that the
# autoregressive property is upheld. natural_ordering=False 
# randomizes autoregressive ordering, while =True makes the autoregressive 
# order p1=f(s_1),p2=f(s_2,s_1)

from models.pytorch_made_master.made import MADE

hidden_layer_sizes=[2*L]

def psi_init(L, hidden_layer_sizes,nout, Form='euler', dtype=torch.float):
    nat_ording=False
    model_r=MADE(L,hidden_layer_sizes, nout, \
                 num_masks=1, natural_ordering=nat_ording)
    model_i=MADE(L,hidden_layer_sizes, nout, \
                 num_masks=1, natural_ordering=nat_ording)

    ppsi=Psi(model_r,model_i, L, form=Form,dtype=datatype, autoregressive=True)
    
    return ppsi

'''############### Autoregressive Sampling and Psi ########################'''

# Begin the autoregressive sampling and Psi forward pass routine 

def Autoregressive_pass(ppsi,s,evals):
    outc=ppsi.complex_out(s) # the complex output given an ansatz form  
    
    # The structure for the NADE is a bit different, assumed binary and doesn't have
    # a variable number of outputs. (# inputs=#outputs). Combining like this 
    # works with current code and the original staggered MADE structure.
    real_probs = torch.cat((ppsi.real_comp(s),1-ppsi.real_comp(s)), dim=1)
    imag_probs = torch.cat((ppsi.imag_comp(s),1-ppsi.imag_comp(s)), dim=1)
    outc= np.exp(real_probs.detach().numpy()+1j*imag_probs.detach().numpy())
    
    new_s=torch.zeros_like(s)
    #   new_s=torch.zeros([N_samples,L])
    
    if len(s.shape)==2:
        [N_samples,L]=s.shape
        nout=outc.shape[1]
    else:
        [N_samples,L]=1,s.shape[0]
        nout=outc.shape[0]
        outc, new_s=outc[None,:], new_s[None,:] # extra dim for calcs
    
    nevals=len(evals)
    
    # Making sure it is an autoregressive model
    assert nout%L==0,"(Output dim)!=nevals*(Input dim), not an Autoregressive NN"
            
    # the full Psi is a product of the conditionals, making a running product easy
    Ppsi=np.ones([N_samples],dtype=np.complex128) # if multiplying
    #Ppsi=np.zeros([N_samples],dtype=np.complex128)  # if adding logs
    
    for ii in range(0, L): # loop over lattice sites
    
        # normalized probability/wavefunction
        vi=outc[:,ii::L] 
        si=s[:,ii] # the input/chosen si (maybe what I'm missing from prev code/E calc)
        if ii==0: # initially have to generate a random set
#            si=torch.tensor(np.random.choice(evals,N_samples),dtype=datatype)
            si=s[:,ii]
        else: # after i=0, the following sampling depends on the previous sample (autoregressive prop)
#            si=new_s[:,ii-1]
            si=s[:,ii]
    #    vi=ppsi.complex_out(si) # the forward pass requires full L inputs, this is 
                                 # not possible, only gives partial information.
                                 # Shouldn't be the case, is autoregressive after all...
        # The MADE is prob0 for 0-nin outputs and then prob1 for 
        # nin-2nin outputs, etc. until ((nevals-1)-nevals)*nin outputs 
    #    tester=np.arange(0,nout);  # print(tester[ii:nlim:L]) # to see slices 
    #    assert len(tester[ii::L])==nevals, "Network Output missing in calculation"
        
        exp_vi=np.exp(vi) # unnorm prob of evals 
        norm_const=np.sqrt(np.sum(np.power(np.abs(exp_vi),2),1))
        psi=np.einsum('ij,i->ij', exp_vi, 1/norm_const) 
        
        born_psi=np.power(np.abs(psi),2)
        
        # satisfy the normalization condition?
        assert np.all(np.sum(born_psi,1)-1<1e-6), "Psi not normalized correctly"
    
        # Now let's sample from the binary distribution
        rands=np.random.rand(N_samples)
        
        psi_s=np.zeros(N_samples, complex) # needed to accumulate Ppsi
        vi_s=np.zeros(N_samples, complex) # accumulate vi for ln(Ppsi)
        checker=np.zeros(N_samples)
        
        for jj in range(nevals): 
            
            prev_selection=(si.numpy()==evals[jj]) # which s were sampled 
            # psi(s), accumulate psi for the s that were used to gen samples
            psi_s+=prev_selection*1*psi[:,jj]
            vi_s+=prev_selection*1*vi[:,jj]
            
            born_s=np.power(np.abs(prev_selection*1*psi[:,jj]),2)
            nonzero_inds=born_s.nonzero()[0]
        
            rands=np.random.rand(len(nonzero_inds))
        
    #       positive condition for eval0
            sel0=(rands<born_s[nonzero_inds])
    #       positive condition for eval1
            sel1=(rands>born_s[nonzero_inds])
            
            # This way of sampling may always lead to a uniform dist.
            new_s[nonzero_inds,ii]= torch.tensor(sel0*1*evals[0]+sel1*1*evals[1],dtype=datatype)
            
            # sampling if a<born_psi, sample
    #        selection=((0<=rands)*(rands-born_psi[:,jj]<1.5e-7)) 
    #        selection=((0<=rands)*(rands-prev_selection*1*born_psi[:,jj]<1.5e-14)) 
    #         Due to precision have to use <=1e-7 as errors will occur
    #         when comparing differences of order 1e-8. (see below check)
            
            checker+=prev_selection*1
            
    #        new_s[selection,ii]=evals[jj]
            
    #        rands=rands-born_psi[:,jj] # shifting the rands for the next sampling
    #        rands=rands-prev_selection*born_psi[:,jj]
            
    #        rands=rands-prev_selection*1*born_psi[:,jj]
            
        # doesn't work
    #        selection1=(rands<np.power(np.abs(psi_s),2))
    #        selection2=(rands>np.power(np.abs(psi_s),2))
    #        checker=selection1*1+selection2*1
    #        new_s[selection1,ii]=evals[1]
    #        new_s[selection2,ii]=evals[0]
            
    #        rands=rands-born_psi[:,jj] # shifting the rands for the next sampling
    #        rands=rands-born_psi[prev_selection,jj]
    #        rands=rands-np.power(np.abs(psi_s),2)
        
    #        if not np.all(checker)==1: 
    #            prob_ind=np.where(checker==0)
    #            raise ValueError("N_samples were not sampled. error at: ", \
    #                prob_ind, 'with ', rands[prob_ind], born_psi[prob_ind,:])
        
    #        if ii==0: prev_psi=np.ones_like(psi_s)
    #        acc_psi=psi_s*prev_psi
    #        prev_psi=psi_s
            
        # Let's just directly sample from psi_i(s) to make sure it works.
        # Doesn't even make sense... Why would I have full state space in first psi(s)?
    #        samplepos=np.zeros([N_samples,1]) # record the sampled states
    #        probs=np.power(np.abs(psi_s),2)
    #        
    #        for ii in range(1,len(probs)):
    #            probs[ii]=probs[ii]+probs[ii-1] # accumulate prob ranges for easy 
    #                                            # sampling with 0<alpha<1
    #        for jj in range(N_samples):
    #            a=np.random.rand()
    #            samplepos[jj]=np.sum(probs<a)
        
        
        
        # Accumulating Ppsi, which is psi_1(s)*psi_2(s)...*psi_L(s)
        Ppsi=Ppsi*psi_s    
#    Ppsi=Ppsi+np.log(psi_s) 
#    
#    Ppsi=Ppsi+(vi_s-0.5*(np.log(np.sum(np.power(np.abs(exp_vi),2),1))))
#Ppsi=np.exp(Ppsi) # may be more numerically stable
#    print('\sum_s|Psi(s)|^2', np.sum(np.power(np.abs(Ppsi),2)))
    #        Ppsi=Ppsi+(vi_s-0.5*(np.log(np.sum(np.power(np.abs(exp_vi),2),1))))
        
        # These are all be equivalent methods



    return Ppsi, new_s

'''################## Test Autoregressive Property #########################'''
L=3
nout=len(evals)*L
# get each spin perm
s2=torch.tensor(np.array(list(itertools.product(evals,repeat=L))),dtype=datatype)

ppsi=psi_init(L,hidden_layer_sizes, nout, 'exponential', datatype)

# Joint probabilities
wvf,new_s=Autoregressive_pass(ppsi,s2,evals)

def psi_i(ppsi, sn, ii, prev_psi=None):
    outc=ppsi.complex_out(sn)
    vi=outc[ii::L] # p(s0), p(s1)
    selection=(sn[ii].numpy()==evals)
    exp_vi=np.exp(vi) 
    norm_const=np.sqrt(np.sum(np.power(np.abs(exp_vi),2)))
    if not prev_psi==None:
        psi=(exp_vi[selection]/norm_const)*prev_psi
    else:        
        psi=exp_vi[selection]/norm_const 
    return psi

# computing the conditionals
ii, jj=0, 0 # lattice, sample number
psi0=psi_i(ppsi,s2[jj],ii) # first conditional psi0
psi1=psi_i(ppsi,s2[jj],ii+1)#,prev_psi=psi0) # second conditional psi1
                            # only enter prev_psi for full mult. conditional
if L==3: psi2=psi_i(ppsi,s2[jj],ii+2)
else: psi2=1

# should be equal to the joint probability
psi00_c=psi0*psi1*psi2
print(wvf[0]-psi00_c)

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

''' Alrighty, let's try the QNADE model I wrote (with some inspiration from the other libs of course) '''

from qnade import QNADE
ppsi = QNADE(L, L, [-1,1])

s2=np.array(list(itertools.product(evals,repeat=L)))
#wvf, new_s = Autoregressive_pass(ppsi,torch.tensor(s2,dtype=datatype),evals)
wvf, _ = ppsi(x=torch.tensor(s2,dtype=datatype))

plt.figure()
plt.bar(range(0,len(s2)), abs(wvf)**2)
plt.title('Probability distribution, |Psi(s)|^2')

N_samp_list = np.logspace(3,7,20)
N_resamples=1 # doesn't really seem to help, default should be 1
#N_samp_list=[1000000]
avg_rel_err = np.zeros([len(N_samp_list),1])
direct_sampling_err = np.zeros([len(N_samp_list),1])
for kk in range(len(N_samp_list)):
    N_samples=int(round(N_samp_list[kk]))
#    s0=torch.tensor(np.random.choice(evals,[N_samples,L]),dtype=datatype)
#    s=s0
#    for ll in range(N_resamples):
#        _,new_s=Autoregressive_pass(ppsi,s,evals)
#        samp_wvf,s = Autoregressive_pass(ppsi,new_s,evals)
    
    alt_wvf, s = ppsi(N_samples)
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

ppsi=psi_init(L,hidden_layer_sizes,len(evals)*L,'exponential')
# get each spin perm
s2=np.array(list(itertools.product(evals,repeat=L)))

wvf,new_s=Autoregressive_pass(ppsi,torch.tensor(s2,dtype=datatype),evals)
wvf=wvf[:,None]

E_sx=np.matmul(np.matmul(np.conjugate(wvf.T),H_sx),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))
E_szsz=np.matmul(np.matmul(np.conjugate(wvf.T),H_szsz),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))
E_tot=np.matmul(np.matmul(np.conjugate(wvf.T),H_tot),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))

N_samples=10000
s0=torch.tensor(np.random.choice(evals,[N_samples,L]),dtype=datatype)
_,new_s=Autoregressive_pass(ppsi,s0,evals)
start=time.time()
_,s = Autoregressive_pass(ppsi,new_s,evals)
end=time.time(); print(end-start)
H_nn=ppsi.O_local(nn_interaction,s.numpy())
H_b=ppsi.O_local(b_field,s.numpy())

print('For psi= \n', wvf, '\n\n the energy (using exact H) is: ', E_tot, '\n while that ' \
      'predicted with the O_local function is: ', np.sum(np.mean(H_b+H_nn,axis=0)), \
      '\n\n for the exact Sx H: ', E_sx, ' vs ',np.sum(np.mean(H_b,axis=0)), \
      '\n\n for exact SzSz H: ', E_szsz ,' vs ', np.sum(np.mean(H_nn,axis=0)))

''' Ensuring that <psi|H|psi> = \sum_s |psi(s)|^2 e_loc(s)   '''

H_szsz_ex=ppsi.O_local(nn_interaction,s2)
H_sz_ex=ppsi.O_local(b_field,s2)
O_loc_analytic= np.sum(np.matmul((np.abs(wvf.T)**2),(H_szsz_ex+H_sz_ex)))\
 /(np.matmul(np.conjugate(wvf.T),wvf))
E_exact=np.matmul(np.matmul(np.conjugate(wvf.T),H_tot),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))

print('\n\n Energy using O_local in the analytical expression: ',O_loc_analytic, \
      '\n vs. that calculated with matrices: ', E_exact )

'''################## Test D v_i(s_i)/DW Calculation #########################'''

ppsi=psi_init(L,hidden_layer_sizes,len(evals)*L,'exponential')
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
# WORKS! Need to modify algorithm

# Now calculate the numerical dvi and compare to analytical expression
outc0=original_net.complex_out(s)
outc1=ppsi.complex_out(s)
vi_dif=(outc1-outc0)/dw # test the difference between vis

test=dpsi*outc0 # dvi/dw (vi_dif) should equal dpsi/dw*vi (for exponential form)
print('\n numerical vi difference: \n', vi_dif, '\n vi*dpsi: \n', test)

'''### Test second term, -1/2 ln(sum_s' |e^(v_i(s'))| deriv ###'''

ii=1
# Now calculate the numerical second term dif for a given i
vi0=outc0[:,ii::L]
vi1=outc1[:,ii::L]
dvi=dpsi[:,ii::L]*vi0
selection=(s[:,ii]==evals[1])*1 # 0 if s=-1 and 1 if s=1
    
sec0=0.5*np.log(np.sum(np.exp(2*np.real(vi0))))
sec1=0.5*np.log(np.sum(np.exp(2*np.real(vi1))))

st_dif=(sec1-sec0)/dw 

exp_t=np.exp(2*np.real(vi0))
norm_term=np.sum(exp_t)
st_test=np.sum(exp_t*dvi)/norm_term # WORKS!!!

print('\n numerical deriv for i=', ii, ' : ', st_dif, \
      '\n analytical deriv for i=', ii, ' : ', st_test)

'''################## Test Full D ln(Psi)/DW Calculation ###################'''

# Now let's expand this dvi calculation to pytorch (test pytorch alg for Dpsi term 1)
# Get my list of vis
outc=original_net.complex_out(s)
out_0=original_net.real_comp(s)
pars=list(original_net.real_comp.parameters())

Ok=np.zeros([np.shape(s)[0]],dtype=complex)
ft_analytic=np.zeros([np.shape(s)[0]],dtype=complex)
st_analytic=np.zeros([np.shape(s)[0]],dtype=complex)
# Accumulate O_omega1 over lattice sites (also have to see which s where used)
for ii in range(0, L): # loop over lattice sites
    vi=outc[:,ii::L] 
    psi_i=out_0[:,ii::L]
    si=s[:,ii].numpy() # the input/chosen si (what I was missing from prev code/E calc)
    
    # pars[0] as we're just looking at a change in 0
    grad0=np.zeros([len(evals),1])
    for kk in range(len(evals)): # have to get the dpsi separately FROM EACH OUTPUT
#        ppsi.real_comp.zero_grad()
#        (out_0[:,(ii+L*kk)]).backward(retain_graph=True) 
        psi_i[:,kk].backward(retain_graph=True) 
        autograd_hacks.compute_grad1(original_net.real_comp)
        autograd_hacks.clear_backprops(original_net.real_comp) 
        grad0[kk]=pars[0].grad1[0,0,0].numpy() 
    
    # The d_w v_i term (just using exponential form for now)
    dvi=vi*grad0.T # for a single param should be of size=#evals
   
    exp_t=np.exp(2*np.real(vi))
    norm_term=np.sum(exp_t)
    sec_term=np.sum(exp_t*dvi)/norm_term # WORKS!!!
    
    temp_Ok=np.zeros([np.shape(s)[0]],dtype=complex)
    for jj in range(len(evals)): 
        
#        selection=(si==evals[jj]) # which s were sampled 
                                #(which indices correspond to the si)
        selection=(s[:,ii]==evals[jj])
        
        sel1=(selection*1).numpy()
        
        # For each eval/si, we must select only the subset vi(si) 
        temp_Ok[:]+=(sel1*dvi[:,jj])
                
    ft_analytic+=temp_Ok
    st_analytic+=sec_term # manual sum over lattice sites ii=0->N
    Ok+=temp_Ok-sec_term

# Calculate the new and old wavefunctions for this s, numerical dln(Psi)
original_net.Autoregressive_pass(s,evals)
wvf0=original_net.wvf
ppsi.Autoregressive_pass(s,evals)
wvf1=ppsi.wvf
wvf_dif=(np.log(wvf1)-np.log(wvf0))/dw; print('\n Numerical dln(Psi)/dw: ', wvf_dif)
print('\n Pytorch analytic dln(Psi)/dw: ', Ok)
print('\n Ratio numerical/analytic: ', np.real(wvf_dif)/np.real(Ok))

deriv=(torch.mean(out_1)-torch.mean(out_0))/dw

print('\n\n numerical deriv dpsi1/dw: ', deriv.item(), '\n pytorch deriv: ', grad0[0][0].item(), \
        '\n ratio: ', deriv.item()/grad0[0][0].item() )

# Peice of Ln(Psi)
out0=original_net.complex_out(s).squeeze()
out1=ppsi.complex_out(s).squeeze()

grads=pars[0].grad1.detach().numpy()
dvi0=0; vi0_l=np.zeros([L,1],dtype=complex)
vs0=0; vs1=0;
sec0=0; sec1=0;
for ii in range(L):
    vi0=out0[ii::L]
    vi1=out1[ii::L]
    
    selection=(s[:,ii]==evals[1])*1 # 0 if s=-1 and 1 if s=1
    
    vs0+=vi0[selection]
    vs1+=vi1[selection]
    
    dvi0+=pars[0].grad1[0][0][0].numpy() #vi0[selection]*
    vi0_l[ii]=vi0[selection]
    
    sec0+=0.5*np.log(np.sum(np.power(np.abs(np.exp(vi0)),2)))
    sec1+=0.5*np.log(np.sum(np.power(np.abs(np.exp(vi1)),2)))
    
ft_diff=(vs1-vs0)/dw; st_diff=(sec1-sec0)/dw
# Making sure things all add up correctly:
assert abs(np.log(wvf1)-(vs1-sec1))<1e-6 and abs(np.log(wvf0)-(vs0-sec0))<1e-6
# this is ensuring that ln(Psi)=prod(psi_s) = sum( vi(si) - 1/2 ln(sum_s'(e^vi))) 

print('\n\n Numerical dPsi/dw minus first-second term diff', wvf_dif-(ft_diff-st_diff))

print('\n Analytic/Pytorch first term: ', ft_analytic)
print('\n First term deriv/difference: ', ft_diff)
print('\n Analytic/Pytorch second term: ', st_analytic)
print('\n Second term deriv/difference: ', st_diff)
print(st_analytic/st_diff)

'''######### Test Full Energy Gradient (over a sample set) #################'''

# function to apply multipliers to the expectation value O_local expression 
def Exp_val(mat,wvf):
    if len(np.shape(mat))==0:
        O_l= np.sum(mat*np.abs(wvf.T)**2)\
        /(np.matmul(np.conjugate(wvf.T),wvf))
    else:
        O_l= np.sum(np.matmul((np.abs(wvf.T)**2),mat))\
        /(np.matmul(np.conjugate(wvf.T),wvf))
    return O_l

par_ind1=0
par_ind2 = 0
par_ind3 =0  

#def autoreg_egrad(par_ind1,par_ind2,par_ind3):
  
ppsi=psi_init(L,hidden_layer_sizes,len(evals)*L,'exponential')
original_net=copy.deepcopy(ppsi)

ppsi.real_comp.zero_grad()

pars1=list(ppsi.real_comp.parameters())

dw=0.001 # sometimes less accurate when smaller than 1e-3
with torch.no_grad():
    if len(pars1[par_ind1].shape)>1:
        pars1[par_ind1][par_ind2][par_ind3]=pars1[par_ind1][par_ind2][par_ind3]+dw
    else:
        print('\n third parameter index ignored as changed parameter set is a vector/bias \n')
        pars1[par_ind1][par_ind2]=pars1[par_ind1][par_ind2]+dw

original_net.Autoregressive_pass(torch.tensor(s2,dtype=torch.float),evals)
wvf0=original_net.wvf
ppsi.Autoregressive_pass(torch.tensor(s2,dtype=torch.float),evals)
wvf1=ppsi.wvf
E_tot0=np.matmul(np.matmul(np.conjugate(wvf0.T),H_tot),wvf0)/(np.matmul(np.conjugate(wvf0.T),wvf0))
E_tot1=np.matmul(np.matmul(np.conjugate(wvf1.T),H_tot),wvf1)/(np.matmul(np.conjugate(wvf1.T),wvf1))
dif=(E_tot1-E_tot0)/dw

print(wvf1-wvf0)

# Get my list of vis
outc=original_net.complex_out(torch.tensor(s2,dtype=torch.float))
if not hasattr(original_net.real_comp,'autograd_hacks_hooks'):             
    autograd_hacks.add_hooks(original_net.real_comp)
outr=original_net.real_comp(torch.tensor(s2,dtype=torch.float))
pars=list(original_net.real_comp.parameters())

Ok=np.zeros([np.shape(s2)[0]],dtype=complex)
# Accumulate O_omega1 over lattice sites (also have to see which s where used)
for ii in range(0, L): # loop over lattice sites
    N_samples=s2.shape[0]
    vi=outc[:,ii::L] 
    psi_i=outr[:,ii::L]
    si=s2[:,ii] # the input/chosen si (what I was missing from prev code/E calc)
    
    # pars[0] as we're just looking at a change in 0
    grad0=np.zeros([N_samples,len(evals)])
    for kk in range(len(evals)): # have to get the dpsi separately FROM EACH OUTPUT vi
#        ppsi.real_comp.zero_grad()
        psi_i[:,kk].mean().backward(retain_graph=True) # mean necessary over samples
                                            # grad1 will save the per sample grad
        autograd_hacks.compute_grad1(original_net.real_comp)
        autograd_hacks.clear_backprops(original_net.real_comp) 
        grad0[:,kk]=pars[par_ind1].grad1[:,par_ind2,par_ind3].numpy() 
    
    # The d_w v_i term (just using exponential form for now)    
    dvi=vi*grad0 # for a single param should be of size=#evals

    exp_t=np.exp(2*np.real(vi))
    norm_term=np.sum(exp_t,1)
    sec_term=np.sum(exp_t*dvi,1)/norm_term 
   
    temp_Ok=np.zeros([np.shape(s2)[0]],dtype=complex)
    for jj in range(len(evals)): 
        
        selection=(si==evals[jj]) # which s were sampled 
                                #(which indices correspond to the si)
        sel1=selection*1
        
        # For each eval/si, we must select only the subset vi(si) 
        temp_Ok[:]+=(sel1*dvi[:,jj])#-sel1*sec_term)
    Ok+=temp_Ok-sec_term # manual sum over lattice sites ii=0->N

[H_nn_ex, H_b_ex]=original_net.O_local(nn_interaction,s2),original_net.O_local(b_field,s2)
E_loc=np.sum(H_nn_ex+H_b_ex,axis=1)

deriv_E0=Exp_val(np.conj(Ok)*E_loc,wvf0)+Exp_val(Ok*np.conj(E_loc),wvf0)-\
Exp_val(E_loc,wvf0)*(Exp_val(np.conj(Ok),wvf0)+Exp_val(Ok,wvf0))

print('\n Expecation val deriv: ', deriv_E0, '\n vs numerical wvf energy diff: ', dif)
print(dif/deriv_E0)

#    return 

''' ############### Gradient for all pars at once ####################### '''

ppsi=psi_init(L,hidden_layer_sizes,len(evals)*L,'exponential')
original_net=copy.deepcopy(ppsi)

comp='imag'
if comp=='real':
    model=original_net.real_comp
else: model=original_net.imag_comp

# Get my list of vis
outc=original_net.complex_out(torch.tensor(s2,dtype=torch.float))
if not hasattr(model,'autograd_hacks_hooks'):             
    autograd_hacks.add_hooks(model)
outr=model(torch.tensor(s2,dtype=torch.float))
pars=list(model.parameters())

# initializing some numpy lists to record both the Ok and be a temporary holder
# for the grad of each eval (which changes each ii, kk loop, but most efficient to initalize once)
N_samples=s2.shape[0]
Ok= [[] for i in range(len(pars))]
gradii= [[] for i in range(len(pars))]
for rr in range(len(pars)):
    if len(pars[rr].size())==2:
        [sz1,sz2]=[pars[rr].size(0),pars[rr].size(1)]
    else:
        [sz1,sz2]=[pars[rr].size(0),1]
    Ok[rr]=np.zeros([N_samples,sz1,sz2],dtype=complex)
    gradii[rr]=np.zeros([N_samples,sz1,sz2,len(evals)])
    
# Accumulate O_omega1 over lattice sites (also have to see which s where used)
for ii in range(0, L): # loop over lattice sites
    N_samples=s2.shape[0]
    vi=outc[:,ii::L] 
    psi_i=outr[:,ii::L]
    si=s2[:,ii] # the input/chosen si (what I was missing from prev code/E calc)
    exp_t=np.exp(2*np.real(vi))
    norm_term=np.sum(exp_t,1)
            
    for kk in range(len(evals)): # have to get the dpsi separately FROM EACH OUTPUT vi
#        original_net.real_comp.zero_grad()
        psi_i[:,kk].mean().backward(retain_graph=True) # mean necessary over samples
                                            # grad1 will save the per sample grad
        autograd_hacks.compute_grad1(model)
        autograd_hacks.clear_backprops(model) 
        for rr in range(len(pars)):
            if len(pars[rr].size())==1:
                gradii[rr][...,kk]=pars[rr].grad1.numpy()[...,None]
            else:
                gradii[rr][...,kk]=pars[rr].grad1.numpy()
                
#        grad0[:,kk]=pars[par_ind1].grad1[:,par_ind2,par_ind3].numpy() 
    for rr in range(len(pars)): # have to include all pars 
        grad=gradii[rr]
    
        # derivative term (will differ depending on ansatz 'form')
        if comp=='real': dvi = np.einsum('il,ijkl->ijkl', vi, grad)
        else:  dvi = np.einsum('il,ijkl->ijkl', 1j*vi, grad)

        st_mult =  np.sum(np.einsum('il,ijkl->ijkl', exp_t, np.real(dvi)),-1)
        sec_term=np.einsum('i,ijk->ijk', 1/norm_term, st_mult)
       
        temp_Ok=np.zeros_like(sec_term,dtype=complex)
        for kk in range(len(evals)): 
            
            selection=(si==evals[kk]) # which s were sampled 
                                        #(which indices correspond to the si)
            sel1=selection*1
                
                # For each eval/si, we must select only the subset vi(si) 
            temp_Ok[:]+=np.einsum('i,ijk->ijk',sel1,dvi[...,kk])
            
        Ok[rr]+= temp_Ok-sec_term # manual sum over lattice sites ii=0->N

# Little algorithm to test the above Ok construction. Will compare the above 
# fully computed Ok for any legitimate combination of parameter indices (i.e.  
# unmasked or existant (bias layers will not have a third parameter index)). 
def Ok_test(par_ind1,par_ind2,par_ind3,comp, dw=0.001):
    ppsi=copy.deepcopy(original_net) # just so it's an equivalent starting spot each time
    if comp=='real': pars1=list(ppsi.real_comp.parameters())
    else: pars1=list(ppsi.imag_comp.parameters())
        
#    dw=0.0001 # sometimes less accurate when smaller than 1e-3
    with torch.no_grad():
        if len(pars1[par_ind1].shape)>1:
            pars1[par_ind1][par_ind2][par_ind3]=pars1[par_ind1][par_ind2][par_ind3]+dw
        else:
            print('\n third parameter index ignored as changed parameter set is a vector/bias \n')
            pars1[par_ind1][par_ind2]=pars1[par_ind1][par_ind2]+dw
    
    original_net.Autoregressive_pass(torch.tensor(s2,dtype=torch.float),evals)
    wvf0=original_net.wvf
    ppsi.Autoregressive_pass(torch.tensor(s2,dtype=torch.float),evals)
    wvf1=ppsi.wvf
    E_tot0=np.matmul(np.matmul(np.conjugate(wvf0.T),H_tot),wvf0)/(np.matmul(np.conjugate(wvf0.T),wvf0))
    E_tot1=np.matmul(np.matmul(np.conjugate(wvf1.T),H_tot),wvf1)/(np.matmul(np.conjugate(wvf1.T),wvf1))
    dif=(E_tot1-E_tot0)/dw
    
    [H_nn_ex, H_b_ex]=original_net.O_local(nn_interaction,s2),original_net.O_local(b_field,s2)
    E_loc=np.sum(H_nn_ex+H_b_ex,axis=1)
    
    this_Ok=Ok[par_ind1][:,par_ind2,par_ind3]
    
    deriv_E0=Exp_val(np.conj(this_Ok)*E_loc,wvf0)+Exp_val(this_Ok*np.conj(E_loc),wvf0)-\
    Exp_val(E_loc,wvf0)*(Exp_val(np.conj(this_Ok),wvf0)+Exp_val(this_Ok,wvf0))
    
    print('\n Expecation val deriv: ', deriv_E0, '\n vs numerical wvf energy diff: ', dif)
    print(dif/deriv_E0)
    
    return

dw=0.01;
for rr in range(len(pars)):
    for ss in range(pars[rr].size(0)):
        if len(pars[rr].size())==2:
            for tt in range(pars[rr].size(1)):
                Ok_test(rr,ss,tt,comp,dw)
        else:
            Ok_test(rr,ss,0,comp, dw)    

''' ############ Now Test the Energy Gradient Using Samples################ ''' 
''' Here we need to draw from the actual distribution Psi (i.e. sample) to get an 
accurate expectation value in respect to said Psi. Can't use s2 (each spin permutation)'''

ppsi=psi_init(L,hidden_layer_sizes,len(evals)*L,'exponential')
original_net=copy.deepcopy(ppsi)

N_samples=10000
s0=torch.tensor(np.random.choice(evals,[N_samples,L]),dtype=torch.float)
_,new_s=Autoregressive_pass(original_net,s0,evals) # Energy calc above seems to demonstrate
_,s = Autoregressive_pass(original_net,new_s,evals) # that it helps to double sample
s=s.numpy()

#s=s2; N_samples=s2.shape[0] # uncomment to use all spin permutations 
                #(can include samples with |Psi(s)|^2=0 and mess with accuracy)

[H_nn_ex, H_b_ex]=original_net.O_local(nn_interaction,s),original_net.O_local(b_field,s)
E_loc=np.sum(H_nn_ex+H_b_ex,axis=1)
E_arg=(np.conj(E_loc)-np.conj(np.mean(E_loc)))

# Get my list of vis
outc=original_net.complex_out(torch.tensor(s,dtype=torch.float))
if not hasattr(original_net.real_comp,'autograd_hacks_hooks'):             
    autograd_hacks.add_hooks(original_net.real_comp)
outr=original_net.real_comp(torch.tensor(s,dtype=torch.float))
pars=list(original_net.real_comp.parameters())

# initializing some numpy lists to record both the Ok and be a temporary holder
# for the grad of each eval (which changes each ii, kk loop, but most efficient to initalize once)
E_grad= [[] for i in range(len(pars))]
gradii= [[] for i in range(len(pars))]
for rr in range(len(pars)):
    if len(pars[rr].size())==2:
        [sz1,sz2]=[pars[rr].size(0),pars[rr].size(1)]
    else:
        [sz1,sz2]=[pars[rr].size(0),1]
    E_grad[rr]=np.zeros([sz1,sz2],dtype=complex)
    gradii[rr]=np.zeros([N_samples,sz1,sz2,len(evals)])
    
## Accumulate O_omega1 over lattice sites (also have to see which s where used)
for ii in range(0, L): # loop over lattice sites
    N_samples=s.shape[0]
    vi=outc[:,ii::L] 
    psi_i=outr[:,ii::L]
    si=s[:,ii] # the input/chosen si (what I was missing from prev code/E calc)
    exp_t=np.exp(2*np.real(vi))
    norm_term=np.sum(exp_t,1)
            
    for kk in range(len(evals)): # have to get the dpsi separately FROM EACH OUTPUT vi
#        original_net.real_comp.zero_grad()
        psi_i[:,kk].mean().backward(retain_graph=True) # mean necessary over samples
                                            # grad1 will save the per sample grad
        autograd_hacks.compute_grad1(original_net.real_comp)
        autograd_hacks.clear_backprops(original_net.real_comp) 
        for rr in range(len(pars)):
            if len(pars[rr].size())==1:
                gradii[rr][...,kk]=pars[rr].grad1.numpy()[...,None]
            else:
                gradii[rr][...,kk]=pars[rr].grad1.numpy()
                
    for rr in range(len(pars)): # have to include all pars 
        grad=gradii[rr]
    
        # derivative term (will differ depending on ansatz 'form')
        dvi = np.einsum('il,ijkl->ijkl', vi, grad)

        st_mult =  np.sum(np.einsum('il,ijkl->ijkl', exp_t, np.real(dvi)),-1)
        sec_term=np.einsum('i,ijk->ijk', 1/norm_term, st_mult)
       
        temp_Ok=np.zeros_like(sec_term,dtype=complex)
        for kk in range(len(evals)): 
            
            selection=(si==evals[kk]) # which s were sampled 
                                        #(which indices correspond to the si)
            sel1=selection*1
                
                # For each eval/si, we must select only the subset vi(si) 
            temp_Ok[:]+=np.einsum('i,ijk->ijk',sel1,dvi[...,kk])
            
        E_grad[rr]+= np.mean(np.einsum('i,ijk->ijk', 2*np.real(E_arg), \
              np.real(temp_Ok-sec_term)),0)
        # mean over the samples in the 0th position

#for rr in range(len(pars)):
#    E_grad[rr]=np.mean(np.einsum('i,ijk->ijk', 2*np.real(E_arg), \
#              np.real(Ok[rr])),0)
    
def egrad_test(par_ind1,par_ind2,par_ind3, E_grad, comp='real', dw=0.001):
    ppsi=copy.deepcopy(original_net) # just so it's an equivalent starting spot each time
    if comp.lower()=='real':
        pars1=list(ppsi.real_comp.parameters())
    else: pars1=list(ppsi.imag_comp.parameters())
        
#    dw=0.001 # sometimes less accurate when smaller than 1e-3
    with torch.no_grad():
        if len(pars1[par_ind1].shape)>1:
            pars1[par_ind1][par_ind2][par_ind3]=pars1[par_ind1][par_ind2][par_ind3]+dw
        else:
            print('\n third parameter index ignored as changed parameter set is a vector/bias \n')
            pars1[par_ind1][par_ind2]=pars1[par_ind1][par_ind2]+dw
    
    original_net.Autoregressive_pass(torch.tensor(s2,dtype=torch.float),evals)
    wvf0=original_net.wvf
    ppsi.Autoregressive_pass(torch.tensor(s2,dtype=torch.float),evals)
    wvf1=ppsi.wvf
    E_tot0=np.matmul(np.matmul(np.conjugate(wvf0.T),H_tot),wvf0)/(np.matmul(np.conjugate(wvf0.T),wvf0))
    E_tot1=np.matmul(np.matmul(np.conjugate(wvf1.T),H_tot),wvf1)/(np.matmul(np.conjugate(wvf1.T),wvf1))
    dif=(E_tot1-E_tot0)/dw
    
    deriv_E0 = E_grad[par_ind1][par_ind2,par_ind3]
    
    print('\n Expecation val deriv: ', deriv_E0, '\n vs numerical wvf energy diff: ', dif)
    print(dif/deriv_E0)
    
    return

dw=0.01; pars=list(ppsi.real_comp.parameters())
for rr in range(len(pars)):
    for ss in range(pars[rr].size(0)):
        if len(pars[rr].size())==2:
            for tt in range(pars[rr].size(1)):
                egrad_test(rr,ss,tt,E_grad,'real',dw)
        else:
            egrad_test(rr,ss,0,E_grad,'real',dw)

''' Making the gradient into a function/method '''

def Autoregressive_grad(ppsi, evals, s, E_arg, comp):
    N_samples=s.shape[0]
    if comp.lower()=='real':
        model=ppsi.real_comp
    else: model=ppsi.imag_comp
    
    # Get my list of vis
    outc=ppsi.complex_out(s)
    if not hasattr(model,'autograd_hacks_hooks'):             
        autograd_hacks.add_hooks(model)
    out=model(s)
    pars=list(model.parameters())
    
    # initializing some numpy lists to record both the Ok and be a temporary holder
    # for the grad of each eval (which changes each ii, kk loop, but most efficient to initalize once)
    E_grad= [[] for i in range(len(pars))]
    gradii= [[] for i in range(len(pars))]
    for rr in range(len(pars)):
        if len(pars[rr].size())==2:
            [sz1,sz2]=[pars[rr].size(0),pars[rr].size(1)]
        else:
            [sz1,sz2]=[pars[rr].size(0),1]
        E_grad[rr]=np.zeros([sz1,sz2],dtype=complex)
        gradii[rr]=np.zeros([N_samples,sz1,sz2,len(evals)])
        
    ## Accumulate O_omega1 over lattice sites (also have to see which s where used)
    for ii in range(0, L): # loop over lattice sites
        N_samples=s.shape[0]
        vi=outc[:,ii::L] 
        psi_i=out[:,ii::L]
        si=s[:,ii] # the input/chosen si (what I was missing from prev code/E calc)
        exp_t=np.exp(2*np.real(vi))
        norm_term=np.sum(exp_t,1)
                
        for kk in range(len(evals)): # have to get the dpsi separately FROM EACH OUTPUT vi
    #        original_net.real_comp.zero_grad()
            psi_i[:,kk].mean().backward(retain_graph=True) # mean necessary over samples
                                                # grad1 will save the per sample grad
            autograd_hacks.compute_grad1(model)
            autograd_hacks.clear_backprops(model) 
            for rr in range(len(pars)):
                if len(pars[rr].size())==1:
                    gradii[rr][...,kk]=pars[rr].grad1.numpy()[...,None]
                else:
                    gradii[rr][...,kk]=pars[rr].grad1.numpy()
                    
        for rr in range(len(pars)): # have to include all pars 
            grad=gradii[rr]
        
            # derivative term (will differ depending on ansatz 'form')
            if ppsi.form.lower()=='exponential':
                if comp.lower()=='real': dvi = np.einsum('il,ijkl->ijkl', vi, grad)
                else: dvi = np.einsum('il,ijkl->ijkl', 1j*vi, grad)
            else: raise ValueError('grad for specified form not defined')
    
            st_mult =  np.sum(np.einsum('il,ijkl->ijkl', exp_t, np.real(dvi)),-1)
            sec_term=np.einsum('i,ijk->ijk', 1/norm_term, st_mult)
           
            temp_Ok=np.zeros_like(sec_term,dtype=complex)
            for kk in range(len(evals)): 
                
                selection=(si==evals[kk]) # which s were sampled 
                                            #(which indices correspond to the si)
                sel1=selection*1
                    
                    # For each eval/si, we must select only the subset vi(si) 
                temp_Ok[:]+=np.einsum('i,ijk->ijk',sel1,dvi[...,kk])
                
            E_grad[rr]+= np.mean(np.einsum('i,ijk->ijk', 2*np.real(E_arg), \
                  np.real(temp_Ok-sec_term)),0)
        
    return E_grad

ppsi=psi_init(L,hidden_layer_sizes,len(evals)*L,'exponential')
original_net=copy.deepcopy(ppsi)

N_samples=10000
s0=torch.tensor(np.random.choice(evals,[N_samples,L]),dtype=torch.float)
_,new_s=Autoregressive_pass(original_net,s0,evals) # Energy calc above seems to demonstrate
_,s = Autoregressive_pass(original_net,new_s,evals) # that it helps to double sample
s=s.numpy()

[H_nn_ex, H_b_ex]=original_net.O_local(nn_interaction,s),original_net.O_local(b_field,s)
E_loc=np.sum(H_nn_ex+H_b_ex,axis=1)

s=torch.tensor(s,dtype=torch.float)
E_arg=(np.conj(E_loc)-np.conj(np.mean(E_loc)))
E_grad_re=Autoregressive_grad(original_net,  evals, s, E_arg, 'real')
E_grad_im=Autoregressive_grad(original_net, evals, s, E_arg, 'imag')

pars=list(ppsi.real_comp.parameters()); dw=0.01
for rr in range(len(pars)):
    for ss in range(pars[rr].size(0)):
        if len(pars[rr].size())==2:
            for tt in range(pars[rr].size(1)):
                egrad_test(rr,ss,tt, E_grad_re, 'real', dw)
        else:
            egrad_test(rr,ss,0, E_grad_re, 'real', dw)

pars=list(ppsi.imag_comp.parameters()); dw=0.01
for rr in range(len(pars)):
    for ss in range(pars[rr].size(0)):
        if len(pars[rr].size())==2:
            for tt in range(pars[rr].size(1)):
                egrad_test(rr,ss,tt, E_grad_im, 'imag', dw)
        else:
            egrad_test(rr,ss,0, E_grad_im, 'imag', dw)


'''###########  Test Model in Psi Object & Method Constructions ############'''

# Test Autograd_hacks (works with modification I added that applies masks)
ppsi=psi_init(L,hidden_layer_sizes,nout,'exponential')
if not hasattr(ppsi.real_comp,'autograd_hacks_hooks'):             
    autograd_hacks.add_hooks(ppsi.real_comp)
outr=ppsi.real_comp(s0)
outr.mean().backward()
autograd_hacks.compute_grad1(ppsi.real_comp) #computes grad per sample for all samples
autograd_hacks.clear_backprops(ppsi.real_comp)
p_r=list(ppsi.real_comp.parameters())

# testing to make sure autograd tools construction for MADE matches the gradient
for param in p_r:
    print(torch.max(param.grad-param.grad1.mean(0)))

ppsi=psi_init(L,hidden_layer_sizes,len(evals)*L,'exponential')
original_net=copy.deepcopy(ppsi)

N_samples=10000
s0=torch.tensor(np.random.choice(evals,[N_samples,L]),dtype=torch.float)
_,new_s=Autoregressive_pass(original_net,s0,evals) # Energy calc above seems to demonstrate
_,s = Autoregressive_pass(original_net,new_s,evals) # that it helps to double sample

[H_nn_ex, H_b_ex]=original_net.O_local(nn_interaction,s.numpy()),original_net.O_local(b_field,s.numpy())
E_loc=np.sum(H_nn_ex+H_b_ex,axis=1)

# compare to above method
E_arg=(np.conj(E_loc)-np.conj(np.mean(E_loc)))
E_grad_re=Autoregressive_grad(original_net,  evals, s, E_arg, 'real')
E_grad_im=Autoregressive_grad(original_net, evals, s, E_arg, 'imag')

# test Gradient 
original_net.autoregressive_grad(E_loc, s, evals, 'real')
original_net.autoregressive_grad(E_loc, s, evals, 'imag')
p_r=list(original_net.real_comp.parameters()); p_i=list(original_net.imag_comp.parameters())

for rr in range(len(pars)):
    print(p_r[rr].grad-torch.tensor(np.real(E_grad_re[rr]),dtype=torch.float).squeeze())
    print(p_i[rr].grad-torch.tensor(np.real(E_grad_im[rr]),dtype=torch.float).squeeze())

'''##### Finally, optimize by combining sampling and gradient descent  #####'''

ppsi=psi_init(L,hidden_layer_sizes,nout,'exponential')

# Enter simulation hyper parameters
N_iter=300
N_samples=100000
lr=0.03
real_time_plot=True
exact_energy=False

# make an initial s
s0=torch.tensor(np.random.choice(evals,[N_samples,L]),dtype=torch.float)
s=ppsi.Autoregressive_pass(s0,evals) 

if real_time_plot:
    plt.figure()
    plt.axis([0, N_iter, min_E-0.5, L])
    plt.axhline(y=min_E,color='r',linestyle='-')

energy_n=np.zeros([N_iter,1])
for n in range(N_iter):
           
    if exact_energy and L<=14: # if want to test the energy without sampling
        s=torch.tensor(s2,dtype=torch.float)
        ppsi.Autoregressive_pass(s, evals)
        
        wvf=ppsi.wvf
        E_tot=np.matmul(np.matmul(np.conjugate(wvf.T),H_tot),wvf)\
        /(np.matmul(np.conjugate(wvf.T),wvf))
        
        # Need sampling, as s2 will have low prob states of Psi disproportionately represented
        # Get the energy at each iteration
        [H_nn, H_b]=ppsi.O_local(nn_interaction,s.numpy()),ppsi.O_local(b_field,s.numpy())
        energy_per_sample = np.sum(H_nn+H_b,axis=1)
        energy_n[n]=E_tot
    else:
        s = ppsi.Autoregressive_pass(s, evals) 
        # Get the energy at each iteration
        [H_nn, H_b]=ppsi.O_local(nn_interaction,s.numpy()),ppsi.O_local(b_field,s.numpy())
        energy_per_sample=np.sum(H_nn+H_b,axis=1)
        energy_n[n] = np.real(np.mean(energy_per_sample))
    
    # calculate the energy gradient, updates pars in Psi object
    ppsi.autoregressive_grad(energy_per_sample, s, evals, 'real') # simple gradient descent
    ppsi.autoregressive_grad(energy_per_sample, s, evals, 'imag')
    
#    lr=lr*0.99 # optional operation, reduces lr in simple iterative way
    ppsi.apply_grad(lr) # releases/updates parameters based on grad method (stored in pars.grad)

    if n%10==0:
        print('percentage of iterations complete: ', (n/N_iter)*100)
    
    if real_time_plot:
        if n>=1:
            plt.plot([n-1,n],[energy_n[n-1],energy_n[n]],'b-')
            plt.pause(0.05)
            plt.draw()


''' Combine the Autoregressive pass and its gradient '''

#def Autoregressive_pass(ppsi,s,evals,datatype=torch.double,grad=True):
#    outc=ppsi.complex_out(s) # the complex output given an ansatz form  
#    N_samples=s.shape[0]
#    nevals=len(evals)
#    
#    # Hooks and allocations necessary for the Energy gradient
#    if not hasattr(ppsi.real_comp,'autograd_hacks_hooks'):             
#        autograd_hacks.add_hooks(ppsi.real_comp)
#    if not hasattr(ppsi.real_comp,'autograd_hacks_hooks'):             
#        autograd_hacks.add_hooks(ppsi.imag_comp)
#    outr=ppsi.real_comp(s)
#    outi=ppsi.imag_comp(s)
#    
#    p_r=list(ppsi.real_comp.parameters())
#    p_i=list(ppsi.real_comp.parameters())
#
#    E_grad_r= [[] for i in range(len(p_r))]
#    E_grad_i= [[] for i in range(len(p_i))]
#    grad_r= [[] for i in range(len(p_r))]
#    grad_i= [[] for i in range(len(p_i))]
#    for rr in range(len(p_r)):
#        if len(p_r[rr].size())==2:
#            [sz1,sz2]=[p_r[rr].size(0),p_r[rr].size(1)]
#        else:
#            [sz1,sz2]=[p_r[rr].size(0),1]
#        E_grad_r[rr]=np.zeros([sz1,sz2],dtype=complex)
#        grad_r[rr]=np.zeros([N_samples,sz1,sz2,nevals])
#        
#    for rr in range(len(p_i)):
#        if len(p_i[rr].size())==2:
#            [sz1,sz2]=[p_i[rr].size(0),p_i[rr].size(1)]
#        else:
#            [sz1,sz2]=[p_i[rr].size(0),1]
#        E_grad_i[rr]=np.zeros([sz1,sz2],dtype=complex)
#        grad_i[rr]=np.zeros([N_samples,sz1,sz2,nevals])
#    
#
#    new_s=torch.zeros_like(s, dtype=datatype)
#    if len(s.shape)==2:
#        [N_samples,L]=s.shape
#        nout=outc.shape[1]
#    else:
#        [N_samples,L]=1,s.shape[0]
#        nout=outc.shape[0]
#        outc, new_s=outc[None,:], new_s[None,:] # extra dim for calcs
#    
#    # Making sure it is an autoregressive model
#    assert nout/L==nevals,"(Output dim)!=nevals*(Input dim), not an Autoregressive NN"
#            
#    # the full Psi is a product of the conditionals, making a running product easy
#    Ppsi=np.ones([N_samples],dtype=np.complex128) 
#    
#    for ii in range(0, L): # loop over lattice sites
#        
#        # normalized probability/wavefunction
#        vi=outc[:,ii::L] 
#        si=s[:,ii] # the input/chosen si (maybe what I'm missing from prev code/E calc)
#        # The MADE is prob0 for 0-nin outputs and then prob1 for 
#        # nin-2nin outputs, etc. until ((nevals-1)-nevals)*nin outputs 
#        
#        tester=np.arange(0,nout);  # print(tester[ii:nlim:L]) # to see slices 
#        assert len(tester[ii::L])==nevals, "Network Output missing in calculation"
#        
#        ''' Energy Gradient '''
#        # Psi i for gradient
#        psi_ii_r=outr[:,ii::L]
#        psi_ii_i=outi[:,ii::L]
#        exp_t=np.exp(2*np.real(vi))
#        exp_t_norm=np.sum(exp_t,1)
#
#        for kk in range(len(evals)): # have to get the dpsi separately FROM EACH OUTPUT vi
##            ppsi.real_comp.zero_grad(); ppsi.imag_comp.zero_grad()
#            psi_ii_r[:,kk].mean().backward(retain_graph=True) # mean necessary over samples
#            autograd_hacks.compute_grad1(ppsi.real_comp) # grad1 will save the per sample grad
#            autograd_hacks.clear_backprops(ppsi.real_comp) 
#            for rr in range(len(p_r)):
#                if len(p_r[rr].size())==1:
#                    grad_r[rr][...,kk]=p_r[rr].grad1.numpy()[...,None]
#                else:
#                    grad_r[rr][...,kk]=p_r[rr].grad1.numpy()
#                    
#            psi_ii_i[:,kk].mean().backward(retain_graph=True)        
#            autograd_hacks.compute_grad1(ppsi.imag_comp)
#            autograd_hacks.clear_backprops(ppsi.imag_comp)                     
#            for rr in range(len(p_i)):
#                if len(p_i[rr].size())==1:
#                    grad_i[rr][...,kk]=p_i[rr].grad1.numpy()[...,None]
#                else:
#                    grad_i[rr][...,kk]=p_i[rr].grad1.numpy()
#        
#        for rr in range(len(pars)): # have to include all pars 
#            grad=gradii[rr]
#    
#            # derivative term (will differ depending on ansatz 'form')
#            dvi = np.einsum('il,ijkl->ijkl', vi, grad)
#    
#            st_mult =  np.sum(np.einsum('il,ijkl->ijkl', exp_t, dvi),-1)
#            sec_term=np.einsum('i,ijk->ijk', 1/norm_term, st_mult)
#           
#            temp_Ok=np.zeros_like(sec_term,dtype=complex)
#            for kk in range(len(evals)): 
#                
#                selection=(si==evals[kk]) # which s were sampled 
#                                            #(which indices correspond to the si)
#                sel1=selection*1
#                    
#                    # For each eval/si, we must select only the subset vi(si) 
#                temp_Ok[:]+=np.einsum('i,ijk->ijk',sel1,dvi[...,kk])
#                
#            E_grad[rr]+= np.mean(np.einsum('i,ijk->ijk', 2*np.real(E_arg), \
#                  np.real(temp_Ok-sec_term)),0)
#        
#        ''' sampling and ppsi '''
#        exp_vi=np.exp(vi) # unnorm prob of evals 
#        norm_const=np.sqrt(np.sum(np.power(np.abs(exp_vi),2),1))
#        psi=np.einsum('ij,i->ij', exp_vi, 1/norm_const) 
#        
#        born_psi=np.power(np.abs(psi),2)
#        
#        # satisfy the normalization condition?
#        assert np.all(np.sum(born_psi,1)-1<1e-6), "Psi not normalized correctly"
#    
#        # Now let's sample from the binary distribution
#        rands=np.random.rand(N_samples)
#        
#        psi_s=np.zeros(N_samples, complex) # needed to accumulate Ppsi
#        checker=np.zeros(N_samples)
#        for jj in range(nevals): 
#        
#            prev_selection=(si.numpy()==evals[jj]) # which s were sampled 
#            # psi(s), accumulate psi for the s that were used to gen samples
#            psi_s+=prev_selection*1*psi[:,jj]
#            
#            # sampling if a<born_psi, sample
#            selection=((0<=rands)*(rands-born_psi[:,jj]<=1.5e-7)) 
#            # Due to precision have to use <=1e-7 as errors will occur
#            # when comparing differences of order 1e-8. (see below check)
#            checker+=selection*1
#            
#            new_s[selection,ii]=evals[jj]
#    
#            rands=rands-born_psi[:,jj] # shifting the rands for the next sampling
#        
#        if not np.all(checker)==1: 
#            prob_ind=np.where(checker==0)
#            raise ValueError("N_samples were not sampled. error at: ", \
#                prob_ind, 'with ', rands[prob_ind], born_psi[prob_ind,:])
#                
#        # Accumulating Ppsi, which is psi_1(s)*psi_2(s)...*psi_L(s)
#        Ppsi=Ppsi*psi_s
#
#    return Ppsi, new_s

''' numerical energy grad via sampling ''' 

#s0=torch.tensor(np.random.choice(evals,[N_samples,L]),dtype=torch.float)
#new_s=ppsi.Autoregressive_pass(s0,evals)
#s=ppsi.Autoregressive_pass(new_s,evals)
#
#[H_nn, H_b]=ppsi.O_local(nn_interaction,s.numpy()),ppsi.O_local(b_field,s.numpy())
#E_loc=np.sum(H_nn+H_b,axis=1)
#E0=np.real(np.mean(E_loc))

#autograd_hacks.add_hooks(ppsi.real_comp)
#outr=ppsi.real_comp(s)
#outr.mean().backward()
#autograd_hacks.compute_grad1(ppsi.real_comp)
#
#N_samples=10000
#s0=torch.tensor(np.random.choice(evals,[N_samples,L]),dtype=torch.float)
#_,new_s=Autoregressive_pass(original_net,s0,evals)
#_,s = Autoregressive_pass(original_net,new_s,evals)
#H_nn=original_net.O_local(nn_interaction,s.numpy())
#
#[H_nn_ex, H_b_ex]=original_net.O_local(nn_interaction,s.numpy()),original_net.O_local(b_field,s.numpy())
#E_loc=np.sum(H_nn_ex+H_b_ex,axis=1)
#
## check energy0 estimation to exact energy difference
#print('Energy Relative Error: ', (E_tot0-np.mean(E_loc))/E_tot0)
#
#_,new_s=Autoregressive_pass(ppsi,s0,evals)
#_,s = Autoregressive_pass(ppsi,new_s,evals)
#H_nn=ppsi.O_local(nn_interaction,s.numpy())
#
#[H_nn_ex, H_b_ex]=ppsi.O_local(nn_interaction,s.numpy()),ppsi.O_local(b_field,s.numpy())
#E_loc1=np.sum(H_nn_ex+H_b_ex,axis=1)
#print('Energy Relative Error: ', (E_tot1-np.mean(E_loc1))/E_tot1)
#
#print('O_loc energy difference: ', (np.mean(E_loc1)-np.mean(E_loc))/dw,\
#      '\n compared to wvf diff: ', dif)    


# Get my psi_omega1 gradients (in pars[ii].grad1)
#if not hasattr(original_net.real_comp,'autograd_hacks_hooks'):             
#    autograd_hacks.add_hooks(original_net.real_comp)
#outr=original_net.real_comp(torch.tensor(s2,dtype=torch.float))
#outr.mean().backward()
#autograd_hacks.compute_grad1(original_net.real_comp)
#autograd_hacks.clear_backprops(original_net.real_comp)
#pars=list(original_net.real_comp.parameters())

# Here calculate the base (unaltered) equivalent expression using O_local 
#new_s=original_net.Autoregressive_pass(torch.tensor(s2,dtype=torch.float),evals)
#s=original_net.Autoregressive_pass(new_s,evals)
#[H_nn_ex, H_b_ex]=original_net.O_local(nn_interaction,s.numpy()),original_net.O_local(b_field,s.numpy())

