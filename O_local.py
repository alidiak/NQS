#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:04:45 2020

@author: alex
"""

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import itertools
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

'''################## Class and function definitions, can skip #############'''

class Op:
    
    def __init__(self, matrix):
        self.matrix=matrix
        self.sites=[]
        
    def add_site(self,new_site):
        self.sites.append(new_site)
        
    # I could potentially add the O_loc function here as a method to Op
        
        return

'''###################### Complex Psi ######################################'''

class Psi:
    
    def __init__(self,real_comp,imag_comp,L,form=None):
        # options for form are 'euler' or 'vector' - corresponding to 2 forms 
        # of complex number notation
        self.real_comp=real_comp
        self.imag_comp=imag_comp
        self.complex=0
        self.L=L
        self.samples=0
        if form==None: # setting the default form as magnitude/angle(phase)
            self.form='euler'
        else:
            self.form=form
        
    # Method to return the complex number specified by the state of the 
    # 2 ANNs real_comp and imag_comp and an input state s
    def complex_out(self, s):
        self.complex=np.zeros(s.size(0),dtype=complex) # complex number for each sample
        if self.form.lower()=='euler':
            self.complex=self.real_comp(s).detach().numpy()*    \
            np.exp(1j*self.imag_comp(s).detach().numpy())
        elif self.form.lower()=='vector':
            self.complex=self.real_comp(s).detach().numpy()+    \
            1j*self.imag_comp(s).detach().numpy()
        else:
            raise Warning('Specified form', self.form, ' for complex number is'\
            ' ambiguous, use either "euler": real_comp*e^(i*imag_comp) or vector":'\
            ' "real_comp+1j*imag_comp. This output was calculated using "euler".')
            self.form='euler'
            self.complex=self.real_comp(s).detach().numpy()*    \
            np.exp(1j*self.imag_comp(s).detach().numpy())
        return self.complex

    '''##################### Energy Gradient ############################'''
    ''' This method will apply the energy gradient to each ANN network param for 
    a given form of Psi. It does simple gradient descent (no SR or anything).
    It does so given an E_local, Energy E, and wavefunc Psi over sample set s.'''

    def apply_energy_gradient(self,s,E_loc,E, lr=0.03): # add Pytorch optimizer) (fixed lr for now)
        
        N_samples=s.size(0)
        outr = self.real_comp(s) #.flatten()
        
        E=np.conj(E)
        E_loc=np.conj(E_loc)
        diff=(E_loc-E)
        
        # should be the simpler form to apply dln(Psi)/dw_i
        if self.form.lower()=='euler':
            # each form has a slightly different multiplication form
            with torch.no_grad():
                multiplier = 2*np.real(np.divide(diff[:,None],outr.detach().numpy())) 
                #multiplier = 2*np.real(E_loc)
        elif self.form.lower()=='vector':
            if np.all(self.complex==0):
                self.complex_out(self,s) # define self.complex
            
            with torch.no_grad():
                multiplier = 2*np.real((E_loc-E)/self.complex.flatten())
                #multiplier = 2*np.real(E_loc)
                
        self.real_comp.zero_grad()
        outr.backward(torch.tensor((1/N_samples)*multiplier)*outr.detach())
        # calling this applies autograd to only tensor .grad object i.e. out
        # which corresponds to dpsi_real(s)/dpars. 
        # must include the averaging over # samples factor myself 
        
        params=list(self.real_comp.parameters()) # get the parameters
        
        # for testing purposes
        pr1_grad=params[0].grad
        
        with torch.no_grad():
            for param in params:
                param -= lr*param.grad # apply the Energy gradient descent
        
        ## Now do the same for the imaginary network. note, they are not done 
        # in parallel here as the networks and number of parameters therein can 
        # vary arbitrarily, so the for loop has to be over each ANN separately
        # this way is also less memory intensive as out, mult, etc. are overwritten
        out = self.imag_comp(s).flatten()

        # complex versions of the multiplier 
        if self.form.lower()=='euler':
            #multiplier = 2*np.imag(E_loc)
            multiplier = 2*np.real((E_loc-E)*1j)
            # 2*np.real((E_loc-E)*1j) or 2*np.imag(-E_loc), but gettin sign err
            # note, to accelerate, Energy E could not appear here because it 
            # should be real. Where it is not are sampling/accuracy errors
            
        elif self.form.lower()=='vector':
            # self.complex should have been defined in previous section of method
            with torch.no_grad():
                multiplier = 2*np.real(1j*(E_loc-E)/self.complex.flatten())
                #multiplier = 2*np.imag(E_loc)
                
        self.imag_comp.zero_grad()
        out.backward(torch.tensor((1/N_samples)*multiplier))#*out)
        
        params=list(self.imag_comp.parameters()) # get the parameters
        
        pi1_grad=params[0].grad
        
        with torch.no_grad():
            for param in params:
                param -= lr*param.grad
            
        return pr1_grad, pi1_grad

    '''###################### Sampling function ################################'''
    def sample_MH(self, N_samples, spin=None, evals=None, s0=None, rot=None):
        # need either the explicit evals or the spin 
        if spin is None and evals is None:
            raise ValueError('Either the eigenvalues of the system or the spin\
                             must be entered')
                
        # the rule for flipping/rotating a spin between it's eigenvalues
        if rot is None:
            if spin is None:
                dim=len(evals)
            else:
                dim = int(2*spin+1)
            rot =  2*np.pi/dim # assume a rotation that scales with # evals
            # note, can only rotate to 'intermediate/nearby' evals
    
        if evals is None:
            evals=2*np.arange(-spin,spin+1) # +1 is just so s=spin is included
            # times 2 is just the convention that's been used, spin evals of -1,1
        
        if s0 is None:
            s0=np.random.choice(evals,size=self.L)
        
        self.samples=np.zeros([N_samples,self.L])
        self.samples[0,:]=s0
        for n in range(N_samples-1):
            
            pos=np.random.randint(L) # position to change
            
            alt_state = self.samples[n,:].copy() # next potential state
            
            if np.random.rand()>=0.5:
                alt_state[pos] = np.real(np.exp(1j*rot)*alt_state[pos]) # flip next random position for spin
            else:
                alt_state[pos] = np.real(np.exp(-1j*rot)*alt_state[pos]) # same chance to flip other direction
            # Will have to generalize to complex vals sometime
            
            # Probabilty of the next state divided by the current
            prob = (np.square(np.abs(self.complex_out(torch.tensor(alt_state,dtype=torch.float32)))))   \
            /(np.square(np.abs(self.complex_out(torch.tensor(self.samples[n,:],dtype=torch.float32)))))
            
            A = min(1,prob) # Metropolis Hastings acceptance formula

            if A ==1: self.samples[n+1,:]=alt_state
            else: 
                if np.random.rand()<A: self.samples[n+1,:]=alt_state # accepting move with prob
                else: self.samples[n+1,:] = self.samples[n,:]
            
        return self.samples
    
'''############################ O_local #######################################
Now find O_local where O is an arbitrary operator acting on sites entered. This 
function returns the O_local operator summed over the 'allowed' transitions 
between the given input spin s and any non-zero transition to spin config s'. 
This operator also depends upon the current wavefunction psi. Psi is a Neural 
Network object that itself is fed s. 
'''

def O_local(operator,s, psi): # potential improvement to use all tensor funcs so
    
    # Testing if it is a Hamiltonian object
#    if hasattr(operator,'Op_list'):
#        N_ops=len(operator.Op_list)
#    else:
#        N_ops=1
#        
    #if not np.all(np.conjugate(np.transpose(operator.matrix))==operator.matrix):
    #    raise Warning('Operator matrix ', operator.matrix, 'is not Hermitian,'\
    #                  ' Observable may be non-real and unphysical')
                 # using CUDA devices could potentially accelerate this func?
    sites=operator.sites.copy()
    
    [n_sites,op_span]= np.shape(sites) # get lattice list length and operator span
                                        # former is often equal to L (lat size) if applied to all sites
    [N_samples,L]=np.shape(s)  # Get the number of samples and lattice size from samples
        
    # For now, assuming we are working in the Sz basis with spin=1/2
    spin=0.5 # presumambly will be entered by the user in later versions
    dim = int(2*spin+1)
    evals=2*np.arange(-spin,spin+1)
    # multiplied by 2 simply because integers are easier to work with
    
    op_size=np.log((np.shape(operator.matrix)[0]))/np.log(dim)
    if not op_size==op_span:
        raise ValueError('Operator size ', op_size, ' does not match the number' \
                         ' of sites entered ', op_span, 'to be acted upon')
    
    O_loc=np.zeros([N_samples,L],dtype=complex) 
    #this construction allows us to get local expectation vals
    # and the energy for each sample (which we can use to backprop)
    
    # cycle through the sites and apply each operator to state s_i (x s_i+1...)
    for i in range(n_sites):
        
        s_prime=s.copy() # so it's a fresh copy each loop
        
        # Set up spin config representation in Sz basis for just the acted upon spins
        # need to generalize for arbitrary spin
        sz_basis=np.zeros([N_samples,op_span,dim])
        s_loc=s[:,sites[i]]
        # can iterate over this for spin!=0.5. use evals
        sz_basis[np.where(s_loc==1)[0],np.where(s_loc==1)[1],:]=np.array([1,0]) 
        sz_basis[np.where(s_loc==-1)[0],np.where(s_loc==-1)[1],:]=np.array([0,1])
        
        if op_span>1: # extend the size of the basis for multi-site operators
            basis=sz_basis[:,0,:]
            for j in range(1,op_span): 
                basis=np.einsum('nk,nl->nkl',basis,sz_basis[:,j,:]).reshape(basis.shape[0],-1)
                # einstein summation func, forces kron product over 2nd axis by 
                # by taking matching inputs from the nth col, avoids kron over samples
                # cycle through the states acted on by the multi-site operator
        else:
            basis=sz_basis[:,0,:]
            
        # S[sites[i]] transformed by Op still in extended basis
        # Should not matter which side the operator acts on as Op must be hermitian
        # as we act on the left here, 
        xformed_state=np.squeeze(np.matmul(basis,operator.matrix)) 
        
#        if np.all((basis>0)==(np.abs(xformed_state)>0)):
#            # can skip changing s', nothing's changed in the basis
#            basis[basis==0]=1 # just doing this so that there's no division by 0
#            div=xformed_state/basis
#            multiplier=div[div>0] 
#            # returns all of the differences in magnitude of the otherwise unchanged basis state
#            pass
#        else:
                        
        ''' 
        Decomposing the kronicker product back into the alternate s'
        requires recursively splitting the vector in 1/dim, if the 1 is in  
        the top (bottom) section, the first kronickered index is 1 (0). Keep 
        splitting the vector untill all indices have been reassigned and a 
        vec of size dim is left - which will be the very last site index.
        I assume the states s are ordered where the state with eval spin is 
        (1,0...,0), that with (0,1,0...,0) is spin-1, etc. 
        '''

        # just so the alg can handle single sample input. adds a singleton dim
        if len(xformed_state.shape)==1:
            xformed_state=xformed_state[None,:]

        ## Generating all possible permutations of the local spins
        perms=np.array(list(itertools.product(evals,repeat=op_span)))
        
        # do a loop over all of the possible permutations
        for kk in range(len(perms)): # xformed_state.shape[1]
            
            # change the local spins in s' for each config
            s_prime[:,sites[i]]=perms[-(kk+1)]
            # -(kk+1) is used because the ordering is opposite of how the 
            # slicing is organized. Ex, -1,-1 corresponds to the last 
            # slice (0,0,0,1) and 1,1 to the first (1,0,0,0) with the 
            # 1 state = (1,0) and -1 state = (0,1) convention.
            
            with torch.no_grad():
                O_loc[:,i]+= xformed_state[:,kk]* \
            (psi.complex_out(torch.tensor(s_prime,dtype=torch.float))  \
        /psi.complex_out(torch.tensor(s,dtype=torch.float))).flatten()
            # each slice of the transformed state acts as a multiplier to 
            # its respective local spin configuration state
                
    return O_loc

'''############### End of Class and function definitions ###################'''

'''
entry method inspired by NetKet-1.0.0
'''

# Define operators to use in the Hamiltonian
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.array([[1, 0], [0, -1]])
#sigmay = (1/1j)*np.array([[0,1],[-1,0]])
# imaginary numbers aren't really supported in the code yet

#matricks=np.array([[2,3],[3,4]]) # for the obs to be physical, it must be Hermitian

#rando_mat=np.kron(matricks,2*matricks) 

b=0.5
J=1

szsz = np.kron(sigmaz, sigmaz)
#b_sx=np.kron(b*sigmax,np.eye(2))+np.kron(np.eye(2),b*sigmax)
#szsz = np.kron(np.kron(np.kron(sigmax, sigmax),sigmax),sigmax)

# initiate the operators and the matrix they are fed
nn_interaction=Op(-J*szsz)#+b_sx)
b_field=Op(b*sigmax)
#rando_op=Op(rando_mat)
#play_op=Op(matricks)

L = 3
for i in range(L):  # Specify the sites upon which the operators act
    # specify the arbitrary sites which the operators will act on
    b_field.add_site([i])
    
    # for n body interactions, n sites must be added
    #nn_interaction.add_site([i, (i + 1) % L,(i+2)%L,(i+3)%L]) 
    nn_interaction.add_site([i,(i+1)%L])#    if hasattr(operator,'Op_list'):
#        N_ops=len(operator.Op_list)
#    else:
#        N_ops=1
    
    #    play_op.add_site([i])
#    rando_op.add_site([i,(i+1)%L])
    
# Make sure to re-instanciate the Op class instances otherwise the site lists will keep appending

'''##################### SIMPLE FFNN DEFINITION ###############################
Here is an example of how to use this function with a Pytorch neural network
(here a simple FFNN created with the class Sequential)
'''

H=40 # hidden layer size
N_samples=100

s=np.random.randint(-1,high=1,size=[N_samples,L]); s[s==0]=1; 
s=torch.tensor(s,dtype=torch.float)

# creates an instance of the Sequential class nn.Sigmoid etc usually in forward section
toy_model=nn.Sequential(nn.Linear(L,H),
                        nn.Sigmoid(),
                        nn.Linear(H,1),
                        nn.Sigmoid()) # relu can be an important ending func as it
             #is always positive for which is necessary for applying log later

H2=20
toy_model2=nn.Sequential(nn.Linear(L,H2),
                        nn.Sigmoid(),
                        nn.Linear(H2,1),
                        nn.Sigmoid()) 

# Test complex wavefunction object construction with modulus and angle
ppsi_mod=Psi(toy_model,toy_model2, L, form='euler')

z1=ppsi_mod.complex_out(s)

# or try the network as a simple z=a+bi complex vector representation
ppsi_vec=Psi(toy_model,toy_model2, L, form='vector')

z2=ppsi_vec.complex_out(s)

print('Complex Psi with Euler representation: \n', z1 , '\n\n and with vector rep: \n', z2)

''' Now let's calculate the loss/cost function (energy) using the O_local function '''

H_nn=O_local(nn_interaction,s.numpy(),ppsi_mod)
H_b=O_local(b_field,s.numpy(),ppsi_mod)

# Calculate the total energy (the sum of these operator expectation val terms)
energy_per_sample = np.sum(H_b,axis=1)
energy=np.mean(energy_per_sample)
print('Total energy is: ', energy)

#with torch.no_grad(): # important to use no_grad here so incorrect grad fns aren't propagated
#    energy_per_sample = torch.sum(H_nn+H_b,axis=1)

'''     Applying the gradient of E to each param, simple gradient descent   '''

ppsi_mod.apply_energy_gradient(s,energy_per_sample,energy)
ppsi_vec.apply_energy_gradient(s,energy_per_sample,energy)
# this function applies the energy gradient to both the real and imaginary
# networks defining psi. works for both forms, euler: a*e^bi and vector: a+bi
 
''' Now let's test the Metropolis-Hastings sampling method in the class Psi '''

N_samples=100
samples=ppsi_mod.sample_MH(N_samples,spin=0.5)

'''##################### Testing Opimization Routine #######################'''

# Enter simulation hyper parameters
N_iter=200
N_samples=1000
burn_in=200
lr=0.1

# make an initial s
s=np.random.randint(-1,high=1,size=[N_samples,L]); s[s==0]=1; 
s=torch.tensor(s,dtype=torch.float)

energy_n=np.zeros([N_iter,1])
for n in range(N_iter):
    
    # Get the energy at each iteration
    H_nn=O_local(nn_interaction,s.numpy(),ppsi_mod)
 #   H_b=O_local(b_field,s.numpy(),ppsi_mod)
    energy_per_sample = np.sum(H_nn,axis=1)
    energy=np.mean(energy_per_sample)
    energy_n[n]=np.real(energy)

    # apply the energy gradient, updates pars in Psi object
    ppsi_mod.apply_energy_gradient(s,energy_per_sample,energy,lr)

    # before doing the actual sampling, we should do a burn in
    sburn=ppsi_mod.sample_MH(burn_in,spin=0.5)

    # Now we sample from the state and recast this as the new s, s0 so burn in is used
    s=torch.tensor(ppsi_mod.sample_MH(N_samples,spin=0.5,s0=sburn[-1,:]),dtype=torch.float)

    if n%10==0:
        print('percentage of iterations complete: ', (n/N_iter)*100)

plt.figure
plt.plot(range(N_iter),energy_n)
plt.xlabel('Iteration number')
plt.ylabel('Energy')


'''#################### For testing O_local set L=3 ########################'''
spin=0.5    
evals=2*np.arange(-spin,spin+1)
s=np.array(list(itertools.product(evals,repeat=L)))

wvf=ppsi_mod.complex_out(torch.tensor(s,dtype=torch.float))

S1=np.kron(np.kron(sigmax,np.eye(2)),np.eye(2))
S2=np.kron(np.kron(np.eye(2),sigmax),np.eye(2))
S3=np.kron(np.kron(np.eye(2),np.eye(2)),sigmax)

H_sx=b*(S1+S2+S3)
H_szsz=-J*(np.diag([-3,1,1,1,1,1,1,-3]))
H_tot=H_szsz+H_sx

E_sx=np.matmul(np.matmul(np.conjugate(wvf.T),H_sx),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))
E_szsz=np.matmul(np.matmul(np.conjugate(wvf.T),H_szsz),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))
E_tot=np.matmul(np.matmul(np.conjugate(wvf.T),H_tot),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))

H_nn=O_local(nn_interaction,s,ppsi_mod)
H_b=O_local(b_field,s,ppsi_mod)

print('For psi= \n', wvf, '\n\n the energy (using exact H) is: ', E_tot, '\n while that ' \
      'predicted with the O_local function is: ', np.sum(np.mean(H_b+H_nn,axis=0)), \
      '\n\n for the exact Sx H: ', E_sx, ' vs ',np.sum(np.mean(H_b,axis=0)), \
      '\n\n for exact SzSz H: ', E_szsz ,' vs ', np.sum(np.mean(H_nn,axis=0)))

print('\n\n also compare the predicted energy per sample of -sz*sz to the spins: '\
      'O_local sz*sz energy: \n', H_nn , '\n\n spins: \n', s )

#O_l=np.matmul(np.matmul(np.conjugate(wvf.T),Sx),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))

''' Also checking with SxSx '''
sxsx=Op(np.kron(sigmax, sigmax))#+b_sx)

L = 3
for i in range(L):  # Specify the sites upon which the operators act
    sxsx.add_site([i,(i+1)%L])

H_sxsx=O_local(sxsx,s,ppsi_mod)

S1=np.kron(np.kron(sigmax,sigmax),np.eye(2))
S2=np.kron(np.kron(np.eye(2),sigmax),sigmax)
S3=np.kron(np.kron(sigmax,np.eye(2)),sigmax)

H_exact= S1+S2+S3
E_exact=np.matmul(np.matmul(np.conjugate(wvf.T),H_exact),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))

print('the energy of Sx*Sx (using exact H) is: ', E_exact, '\n with O_l it is: ' \
      ,np.sum(np.mean(H_sxsx,axis=0)) )

''' Set the weights to some arbitrary values '''
        
with torch.no_grad():
    for param in ppsi_vec.real_comp.parameters():
        param.copy_(0.3*torch.ones_like(param))
    for param in ppsi_vec.imag_comp.parameters():
        param.copy_(0.3*torch.ones_like(param))


'''###### Let's start by testing the grad of just the avg. of out ##########'''
N_samples=1
s=np.random.randint(-1,high=1,size=[N_samples,L]); s[s==0]=1; 

K=21
ratios=[]
for i in range(1,K):
    toy_model=nn.Sequential(nn.Linear(L,H),nn.Sigmoid(), nn.Linear(H,1),nn.Sigmoid())
    mult=i/10
    
    with torch.no_grad():
        for param in toy_model.parameters():
            param.copy_(mult*torch.ones_like(param))
    
    out_0 = toy_model(torch.tensor(s,dtype=torch.float))
    out_0.backward(out_0)
    
    pars=list(toy_model.parameters())
    grad0=pars[0].grad #* (1/N_samples)
    dw=0.001
    with torch.no_grad():
        pars[0][0][0]=pars[0][0][0]+dw
        # to change all params by dw
        #for param in toy_model.parameters():
        #    param.copy_(mult*torch.ones_like(param)+dw*torch.ones_like(param))
            
    out_1=toy_model(torch.tensor(s,dtype=torch.float))
    
    deriv=(torch.mean(out_1)-torch.mean(out_0))/dw
    
    print('numberical deriv: ', deriv.item(), '\n pytorch deriv: ', grad0[0][0].item(), \
            '\n ratio: ', deriv.item()/grad0[0][0].item() )
    
    ratios.append(deriv.item()/grad0[0][0].item())

plt.figure()
plt.plot([i/10 for i in range(1,K)],ratios)
# strangely, the accuracy increases with the avg value in the NN
# also, the increasing the # hidden nodes reduces acc. (likely in numerical tho)

'''################### Checking the gradient method ########################'''

N_samples=30
s=np.random.randint(-1,high=1,size=[N_samples,L]); s[s==0]=1; 
## the following gives a gradient of 0 for some reason:
#s=np.ones([N_samples,L])
#s[:,1]=-1


def vec_init(L, H=40, mult=0.3):
    toy_model=nn.Sequential(nn.Linear(L,H),nn.Sigmoid(), nn.Linear(H,1),nn.Sigmoid()) 
    H2=round(H/2)
    toy_model2=nn.Sequential(nn.Linear(L,H2),nn.Sigmoid(),nn.Linear(H2,1),nn.Sigmoid()) 

    ppsi_vec=Psi(toy_model,toy_model2, L, form='vector')
    
    # set the starting param to a certain value so we can test consistency
    with torch.no_grad():
        for param in ppsi_vec.real_comp.parameters():
            param.copy_(mult*torch.ones_like(param))
        for param in ppsi_vec.imag_comp.parameters():
            param.copy_(mult*torch.ones_like(param))
    return ppsi_vec

def mod_init(L, H=40, mult=0.3):
    
    m=nn.Sequential(nn.Linear(L,H),nn.Sigmoid(), nn.Linear(H,1),nn.Sigmoid()) 
    H2=round(H/2)
    m2=nn.Sequential(nn.Linear(L,H2),nn.Sigmoid(),nn.Linear(H2,1),nn.Sigmoid()) 

    
    ppsi_mod=Psi(m,m2, L, form='euler')

    with torch.no_grad():
        for param in ppsi_mod.real_comp.parameters():
            param.copy_(mult*torch.ones_like(param))
        for param in ppsi_mod.imag_comp.parameters():
            param.copy_(mult*torch.ones_like(param))
    return ppsi_mod

# assign the params
veci_params=list(ppsi_vec.imag_comp.parameters())
modr_params=list(ppsi_mod.real_comp.parameters())
modi_params=list(ppsi_mod.imag_comp.parameters())


                
#self.complex_out(self,s) # define self.complex
#multiplier = 2*np.real((E_loc-E)/self.complex.flatten())
#                
#self.real_comp.zero_grad()
#out.backward(torch.tensor((1/N_samples)*multiplier)*out)
#      
#params=list(self.real_comp.parameters()) # get the parameters 
#pr1_grad=params[0].grad
#
#        out = self.imag_comp(s).flatten()
#
#multiplier = 2*np.imag(E_loc)
##            multiplier = 2*np.real((E_loc-E)*1j)
# 
#multiplier = 2*np.real(1j*(E_loc-E)/self.complex.flatten())
#                
#        self.imag_comp.zero_grad()
#        out.backward(torch.tensor((1/N_samples)*multiplier)*out)


# is important to re-initialize before testing for ensuring the gradient isn't applied
H,mult=6,0.1
ppsi_vec=vec_init(L,H,mult)
vecr_params=list(ppsi_vec.real_comp.parameters())

# starting by testing the vector version
del_wr=0.01 # delta w real

# first calculate the energy before changing the parameters
[H_nn, H_b]=O_local(nn_interaction,s,ppsi_vec),O_local(b_field,s,ppsi_vec)
energy=np.sum(np.mean(H_nn+H_b,axis=0)) 
E_loc=np.sum(H_nn+H_b,axis=1)

# compare to the method derived gradient (using the original energy/params)
[p1_r,p1_i]=ppsi_vec.apply_energy_gradient(torch.tensor(s,dtype=torch.float) \
                        ,E_loc,energy,0.03)

# reset the parameter
ppsi_vec=vec_init(L,H,mult)
vecr_params=list(ppsi_vec.real_comp.parameters())

# now change a particular parameter value by delta w=wr+i*wi
with torch.no_grad():
    vecr_params[0][0][0]=vecr_params[0][0][0]+del_wr

[H_nn2, H_b2]=O_local(nn_interaction,s,ppsi_vec),O_local(b_field,s,ppsi_vec)
new_energy=np.sum(np.mean(H_nn2+H_b2,axis=0)) 

# get the numerical derivative/slope
dE=(new_energy-energy)/del_wr

print('Numerical derivative (real vector): ', dE, '\n derivative from method: ' \
      ,p1_r[0][0].item(), '\n difference: ', np.abs(p1_r[0][0].item())-np.abs(dE), \
      '\n ratio dE/method: ', p1_r[0][0].item()/np.real(dE))


# Now for the imaginary comp
H,mult=18,0.5
ppsi_vec=vec_init(L,H,mult)
veci_params=list(ppsi_vec.imag_comp.parameters())

del_wi=0.01 # delta w imag

[H_nn, H_b]=O_local(nn_interaction,s,ppsi_vec),O_local(b_field,s,ppsi_vec)
energy=np.sum(np.mean(H_nn+H_b,axis=0)) 

[p1_r,p1_i]=ppsi_vec.apply_energy_gradient(torch.tensor(s,dtype=torch.float) \
                        ,np.sum(H_nn+H_b,axis=1),energy,0.03)

ppsi_vec=vec_init(L,H,mult)
veci_params=list(ppsi_vec.imag_comp.parameters())

# now change a particular parameter value by delta w=wr+i*wi
with torch.no_grad():
    veci_params[0][0][0]=veci_params[0][0][0]+del_wi

[H_nn2, H_b2]=O_local(nn_interaction,s,ppsi_vec),O_local(b_field,s,ppsi_vec)
new_energy=np.sum(np.mean(H_nn2+H_b2,axis=0)) 

# get the numerical derivative/slope
dE=(new_energy-energy) /del_wi

print('Numerical derivative (imag vector): ', dE, '\n derivative from method: ' \
      ,p1_i[0][0].item(), '\n difference: ', np.abs(p1_i[0][0].item())-np.abs(dE), \
      '\n ratio dE/method: ', p1_i[0][0].item()/np.real(dE))


# With the magnitude version
H,mult=16,0.9
ppsi_mod=mod_init(L,H,mult)
modr_params=list(ppsi_mod.real_comp.parameters())

del_wr=0.01 

[H_nn, H_b]=O_local(nn_interaction,s,ppsi_mod),O_local(b_field,s,ppsi_mod)
energy=np.sum(np.mean(H_nn+H_b,axis=0)) 
E_loc=np.sum(H_nn+H_b,axis=1)

[p1_r,p1_i]=ppsi_mod.apply_energy_gradient(torch.tensor(s,dtype=torch.float) \
                        ,E_loc,energy,0.03)
#ppsi_mod.real_comp.zero_grad()
#out=ppsi_mod.real_comp(torch.tensor(s,dtype=torch.float)).flatten()
#multiplier = 2*np.real((E_loc-energy)/(out.detach().numpy()))   
#out.backward(torch.tensor((1/N_samples)*multiplier)*out)           
#
#params=list(ppsi_mod.real_comp.parameters()) # get the parameters 
#p1_r=params[0].grad

ppsi_mod=mod_init(L,H,mult)
modr_params=list(ppsi_mod.real_comp.parameters())

with torch.no_grad():
    modr_params[0][0][0]=modr_params[0][0][0]+del_wr

[H_nn2, H_b2]=O_local(nn_interaction,s,ppsi_mod),O_local(b_field,s,ppsi_mod)
new_energy=np.sum(np.mean(H_nn2+H_b2,axis=0)) 

dE=(new_energy-energy)/del_wr

print('Numerical derivative (real mod): ', dE, '\n derivative from method: ' \
      ,p1_r[0][0].item(), '\n difference: ', p1_r[0][0].item()-dE, \
      '\n ratio dE/method: ', p1_r[0][0].item()/np.real(dE))

# ppsi_mod.real_comp(torch.tensor(s,dtype=torch.float))

# Now the angle
H,mult=18,0.9
ppsi_mod=mod_init(L,H,mult)
modi_params=list(ppsi_mod.imag_comp.parameters())

del_wi=0.01 

[H_nn, H_b]=O_local(nn_interaction,s,ppsi_mod),O_local(b_field,s,ppsi_mod)
energy=np.sum(np.mean(H_nn+H_b,axis=0)) 
E_loc=np.sum(H_nn+H_b,axis=1)

[p1_r,p1_i]=ppsi_mod.apply_energy_gradient(torch.tensor(s,dtype=torch.float) \
                        ,E_loc,energy,0.03)
#multiplier = 2*np.imag(np.conj(E_loc))
##            multiplier = 2*np.real((E_loc-E)*1j)
#ppsi_mod.imag_comp.zero_grad()
#out=ppsi_mod.imag_comp(torch.tensor(s,dtype=torch.float)).flatten()
#out.backward(torch.tensor((1/N_samples)*multiplier))#)*out)                 
#
#params=list(ppsi_mod.imag_comp.parameters()) # get the parameters 
#p1_i=params[0].grad

ppsi_mod=mod_init(L,H,mult)
modi_params=list(ppsi_mod.imag_comp.parameters())

with torch.no_grad():
    modi_params[0][0][0]=modi_params[0][0][0]+del_wi

[H_nn2, H_b2]=O_local(nn_interaction,s,ppsi_mod),O_local(b_field,s,ppsi_mod)
new_energy=np.sum(np.mean(H_nn2+H_b2,axis=0)) 

dE=(new_energy-energy)/del_wi

print('Numerical derivative (angle): ', dE, '\n derivative from method: ' \
      ,p1_i[0][0].item(), '\n difference: ', p1_i[0][0].item()-dE, \
      '\n ratio dE/method: ', p1_i[0][0].item()/np.real(dE))









''' Unused '''

#class Hamiltonian:
#    
#    def __init__(self,*argv):
#        self.Op_list=[]
#        self.matrix=[]
#        self.sites=[]
#        for arg in argv:
#            self.Op_list.append(arg)
#            self.matrix.append(arg.matrix)
#            self.sites.append(arg.sites)
#
## just enter the operators into Hamiltonian 
#H=Hamiltonian(b_field,nn_interaction)


'''   Could also try to use the energy for each sample as a loss function 
                    (without the explicit energy gradient)             '''

#E_real=torch.tensor(np.real(energy_per_sample), dtype=torch.float32)
## should this be np.abs if form is euler, np.real if vector?
#
#E_imag=torch.tensor(np.imag(energy_per_sample), dtype=torch.float32)
## np.abs if form is euler, np.real if vector?
#
#            ##### Applying to the real/magnitude network #####
#out=toy_model(s).flatten() 
#toy_model.zero_grad()
#out.backward(E_real) 
#
#params=list(toy_model.parameters()) # record the parameters
#with torch.no_grad():
#    for param in params:
#        param -= 0.1 * param.grad
#        
#            ##### Applying to the imag/angle network ######
out=ppsi_vec.imag_comp(torch.tensor(s,dtype=torch.float)).flatten() 
ppsi_vec.imag_comp.zero_grad()
out.backward(out) 

params=list(ppsi_vec.imag_comp.parameters()) # record the parameters
with torch.no_grad():
    for param in params:
        param -= 0.1 * param.grad


'''                 First simple attempted approach,
 this method is inccorrect, it does dln(Ann_w1)/dw1 not dln(Psi(w1,w2))/dw1 '''

#            ##### Applying to the real/magnitude network #####
#out=torch.log(ppsi_mod.real_comp(s)).flatten()
#ppsi_mod.real_comp.zero_grad()
#out.backward(2*E_real*out-2*np.real(energy)*out) 
## seems weird, but Idk how else it'd be done. should be 2*Re(<E_loc*dpsi>-<E><dpsi>)
#
#params=list(ppsi_mod.real_comp.parameters()) # record the parameters
#with torch.no_grad():
#    for param in params:
#        param -= 0.1 * param.grad # modify to be (E_loc-E)*ln(param).grad
#
#            ##### Applying to the imag/angle network ####
#out=torch.log(ppsi_mod.imag_comp(s)).flatten()
#ppsi_mod.imag_comp.zero_grad()
#out.backward(2*E_real*out-2*np.real(energy)*out) 
## seems weird, but Idk how else it'd be done. should be 2*Re(<E_loc*dpsi>-<E><dpsi>)
#
#params=list(ppsi_mod.imag_comp.parameters()) # record the parameters
#with torch.no_grad():
#    for param in params:
#        param -= 0.1 * param.grad      

'''
This is a previously working O_local that calculated the input operator O_loc 
for a single sample at a time. The function was generalized to process a batch 
of samples.
'''
#def O_local(operator,s, psi):  
#    
#    sites=operator.sites.copy()
#    
#    [n_sites,op_span]= np.shape(sites) # get lattice list length and operator span
#                                        # former is often equal to L (lat size)
#        
#    # For now, assuming we are working in the Sz basis with spin=1/2
#    spin=0.5 # presumambly will be entered by the user in later versions
#    dim = int(2*spin+1)
#    # sz_tilde=np.zeros([L,dim**op_span])
#    
#    op_size=np.log((np.shape(operator.matrix)[0]))/np.log(dim)
#    if not op_size==op_span:
#        raise ValueError('Operator size ', op_size, ' does not match the number' \
#                         ' of sites entered ', op_span, 'to be acted upon')
#    
#    O_loc=0
#    # cycle through the sites and apply each operator to state s_i (x s_i+1...)
#    for i in range(n_sites):
#        
#        s_prime=s.copy() # so it's a fresh copy each loop
#        
#        # Set up spin config representation in Sz basis for just the acted upon spins
#        sz_basis=np.zeros([op_span,dim])
#        s_loc=s[sites[i]]
#        sz_basis[np.where(s_loc==1),...]=np.array([1,0]) # need to generalize for arbitrary spin
#        sz_basis[np.where(s_loc==-1),...]=np.array([0,1])
#        
#        if op_span>1: # extend the size of the basis for multi-site operators
#            basis=sz_basis[0]
#            for j in range(1,op_span): 
#                basis=np.kron(basis,sz_basis[j]) 
#                # cycle through the states acted on by the multi-site operator
#        else:
#            basis=sz_basis[0]
#            
#        # S[sites[i]] transformed by Op still in extended basis
#        xformed_state=np.squeeze(np.matmul(basis,operator.matrix)) 
#        
#        if np.all(xformed_state==basis): # can skip changing s', nothing's changed
#            pass
#        else:
#                        
#            ''' 
#            Decomposing the kronicker product back into the alternate s'
#            requires recursively splitting the vector in 1/dim, if the 1 is in  
#            the top (bottom) section, the first kronickered index is 1 (0). Keep 
#            splitting the vector untill all indices have been reassigned and a 
#            vec of size dim is left - which will be the very last site index.
#            I assume the states s are ordered where the state with eval spin is 
#            (1,0...,0), that with (0,1,0...,0) is spin-1, etc. 
#            '''
#            spl=xformed_state.copy()
#            alt_ind=[]
#            while np.max(np.shape(spl))>dim: # need to keep splitting
#            
##            for nn in range(len(xformed_state)/dim):
#                spl=np.split(spl,dim)
#                for kk in range(dim):
#                    if any(spl[kk]):                        
#                        # depending how you set up spin, may need to be changed
#                        alt_ind.append(2*(spin-kk))
#                        # new split section to split again
#                        spl=spl[kk]
#                        break
#            
#            # exiting while loop when = dim, this is the last index
#            alt_ind.append(2*(spin-np.squeeze(np.where(spl==1))))
#                    
#        # now we have identified s'!!
#        # Each local op will only effect op_span number of sites in configuration s
#        s_prime[sites[i]]=alt_ind
#        
#        # Place-holder for wavefunction psi calculation that must be summed over
#        O_loc+=psi(torch.tensor(s_prime))
#        
#    return O_loc


''' ########################## NORMALIZATION ###############################'''

# The simplest method to normalize would just to divide by the sum of the output
#out=out/out.sum() # e.x. for a given 

# actually, I shouldn't need normalization if it's <psi|H|psi>/<psi|psi>



#    def apply_energy_gradient(self,s,E_loc,E, lr=0.03): # add Pytorch optimizer) (fixed lr for now)
#        
#        N_samples=s.size(0)
#        out = self.real_comp(s).flatten()
#        
#        # should be the simpler form to apply dln(Psi)/dw_i
#        if self.form.lower()=='euler':
#            # each form has a slightly different multiplication form
#            with torch.no_grad():
#                multiplier = 2*np.real((E_loc-E)/(out.detach().numpy()))   
#                #multiplier = 2*np.real(E_loc)
#        elif self.form.lower()=='vector':
#            if np.all(self.complex==0):
#                self.complex_out(self,s) # define self.complex
#            
#            with torch.no_grad():
#                multiplier = 2*np.real((E_loc-E)/self.complex.flatten())
#                #multiplier = 2*np.real(E_loc)
#                
#        self.real_comp.zero_grad()
#        out.backward(torch.tensor((1/N_samples)*multiplier)*out)
#        # calling this applies autograd to only tensor .grad object i.e. out
#        # which corresponds to dpsi_real(s)/dpars. 
#        # must include the averaging over # samples factor myself 
#        
#        params=list(self.real_comp.parameters()) # get the parameters
#        
#        # for testing purposes
#        pr1_grad=params[0].grad
#        
#        with torch.no_grad():
#            for param in params:
#                param -= lr*param.grad # apply the Energy gradient descent
#        
#        ## Now do the same for the imaginary network. note, they are not done 
#        # in parallel here as the networks and number of parameters therein can 
#        # vary arbitrarily, so the for loop has to be over each ANN separately
#        # this way is also less memory intensive as out, mult, etc. are overwritten
#        out = self.imag_comp(s).flatten()
#
#        # complex versions of the multiplier 
#        if self.form.lower()=='euler':
#            #multiplier = 2*np.imag(E_loc)
#            multiplier = 2*np.real((E_loc-E)*1j)
#            # 2*np.real((E_loc-E)*1j) or 2*np.imag(-E_loc), but gettin sign err
#            # note, to accelerate, Energy E could not appear here because it 
#            # should be real. Where it is not are sampling/accuracy errors
#            
#        elif self.form.lower()=='vector':
#            # self.complex should have been defined in previous section of method
#            with torch.no_grad():
#                multiplier = 2*np.real(1j*(E_loc-E)/self.complex.flatten())
#                #multiplier = 2*np.imag(E_loc)
#                
#        self.imag_comp.zero_grad()
#        out.backward(torch.tensor((1/N_samples)*multiplier)*out)
#        
#        params=list(self.imag_comp.parameters()) # get the parameters
#        
#        pi1_grad=params[0].grad
#        
#        with torch.no_grad():
#            for param in params:
#                param -= lr*param.grad
#            
#        return pr1_grad, pi1_grad