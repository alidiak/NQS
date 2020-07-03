#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:26:01 2020

Here we keep classes and functions for the Pytorch Quantum States library.


@author: alex
"""

import itertools
import numpy as np
import torch
import copy
import autograd_hacks

class Op:
    
    def __init__(self, matrix):
        self.matrix=matrix
        self.sites=[]
        
    def add_site(self,new_site):
        self.sites.append(new_site)
        
    # Could potentially add the O_loc function here as a method to Op
        
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

    def energy_gradient(self,s,E_loc,E=None): # add Pytorch optimizer) (fixed lr for now)
        
        if E is None:
            E=np.mean(E_loc)
                
        E=np.conj(E)
        E_loc=np.conj(E_loc)
        diff=(E_loc-E)
        
        self.real_comp.zero_grad()
        self.imag_comp.zero_grad()
        # should be the simpler form to apply dln(Psi)/dw_i
        if self.form.lower()=='euler':
            
            outr = self.real_comp(s).flatten()
            outi = self.imag_comp(s).flatten()
            
            # each form has a slightly different multiplication form
            # MODULUS
            mult=torch.tensor(np.real(2*diff),dtype=torch.float)
            (outr.log()*mult).mean().backward()
            # calling this applies autograd to tensor .grad object i.e. out*mult
            # which corresponds to dpsi_real(s)/dpars. 
            
            # ANGLE
            mult = torch.tensor(2*np.imag(-E_loc),dtype=torch.float)
            (mult*outi).mean().backward()
            
        # Although the speed difference is not significant, the above is still 
        # faster than using the autograd_hacks per sample gradient version used
        # for the vector gradients below
            
        elif self.form.lower()=='vector':
            if np.all(self.complex==0): 
        # could create errors if doesn't use the updated ppsi and new s
        # but each call of O_local redefines the .complex
                self.complex_out(s) # define self.complex
              
            # hooks accumulate the gradient per sample into layers.backprops_list
            # only called once otherwise extra grads are accumulated
            if not hasattr(self.real_comp,'autograd_hacks_hooks'):             
                autograd_hacks.add_hooks(self.real_comp)
            if not hasattr(self.imag_comp,'autograd_hacks_hooks'): 
                autograd_hacks.add_hooks(self.imag_comp)
            outr=self.real_comp(s)
            outi=self.imag_comp(s)
            outr.mean().backward()
            outi.mean().backward()
            autograd_hacks.compute_grad1(self.real_comp)
            autograd_hacks.compute_grad1(self.imag_comp)
            
            m=2*(np.conj(E_loc)-np.conj(E))/self.complex.squeeze()
            
            p_r=list(self.real_comp.parameters())
            p_i=list(self.imag_comp.parameters())
            
            # multiplying the base per sample grad in param.grad1 by the dPsi
            # derivative term and assigning to the .grad variable to be applied 
            # to each parameter variable with the apply_grad function. 
            for param in p_r:
                if len(param.size())==2:
                    ein_str="i,ijk->ijk"
                elif len(param.size())==1:
                    ein_str="i,ik->ik"
                param.grad=torch.einsum(ein_str,torch.tensor(np.real(m)\
                    ,dtype=torch.float),param.grad1).mean(0)
            for param in p_i: # dPsi here is 1j*dPsi of real
                if len(param.size())==2:
                    ein_str="i,ijk->ijk"
                elif len(param.size())==1:
                    ein_str="i,ik->ik"
                param.grad=torch.einsum(ein_str,torch.tensor(np.real(1j*m)\
                    ,dtype=torch.float),param.grad1).mean(0)
          
            # clear backprops_list for next run
            autograd_hacks.clear_backprops(self.real_comp)
            autograd_hacks.clear_backprops(self.imag_comp)
            
        return 

    def SR(self,s,E_loc, lambduh=1, cutoff=1e-8): 
        
        E0=np.real(np.mean(E_loc))
    
        if self.form=='vector':
            if np.all(self.complex==0): 
                self.complex_out(s)
            m_r=(1/self.complex).squeeze()
            m_i=1j*m_r
        else:
            m_r=1/self.real_comp(s).detach().numpy().squeeze()
            m_i=(np.ones([s.shape[0],1])*1j).squeeze()
        E_arg=(np.conj(E_loc)-np.conj(E0))
        
        for ii in range(2):
            if ii==0:# Compute SR for real component
                model=self.real_comp; m=m_r
            else:
                model=self.imag_comp; m=m_i
                
            model.zero_grad()
            N_samples=s.shape[0]
            
            if not hasattr(model,'autograd_hacks_hooks'):             
                autograd_hacks.add_hooks(model)
            outr=model(s)
            outr.mean().backward()
            autograd_hacks.compute_grad1(model) #computes grad per sample for all samples
            autograd_hacks.clear_backprops(model)
            pars=list(model.parameters())
                
            for param in pars:
                with torch.no_grad():
                    if len(param.size())==2:#different mat mul rules depending on mat shape
                        ein_str="i,ijk->ijk"
                    elif len(param.size())==1:
                        ein_str="i,ik->ik"
                    if len(param.size())>1:
                        if param.size(1)>param.size(0): # pytorch flips matrix pattern sometimes
        # have to manually flip it back. (else get S=scalar for Nx1 matrix xforms-can't be right)
                            param.grad1=param.grad1.view(param.grad1.size(0),param.size(1),param.size(0))
                    Ok=np.einsum(ein_str,m,param.grad1.numpy())
                    if len(np.shape(Ok))==2:
                        Ok=Ok[:,:,None] 
        # Vector bias values do not agree with original method if this is not present
        # When present though, returns values similar in order to the other param grad values...
                    Exp_Ok=np.mean(Ok,0) # conj(mean)=mean(conj)
        #T1=np.tensordot(np.conj(Ok_list[kk]),Ok_list[kk].T, axes=((0,2),(2,0)))/N_samples
                    T1=np.einsum("kni,imk->nm",np.conj(Ok),Ok.T)/N_samples
                # These are methods are equivalent! Good sanity check (einsum more versitile)
                    St=2*np.real(T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)) # the S+c.c. term
         # In NKT, cutoff of 1e-10 is used for S before inverting - inc. numerical stability?
         # if diagS< cutoff, S(i,i)=1 and s.row=s.col=0. 
                    St[St<cutoff]=0
                    l_reg=lambduh*np.eye(St.shape[0],St.shape[1])*np.diag(St) # regulation term
                    S_inv=torch.tensor(np.linalg.pinv(St+l_reg),dtype=torch.float) # S^-1 term with reg
                    force=torch.einsum(ein_str,torch.tensor(np.real(2*E_arg*m)\
                    ,dtype=torch.float),param.grad1).mean(0) # force/DE term
                    # Compute SR 'gradient'
                    if len(force.size())==1:  # deal with .mm issues when vector Mx1
                        param.grad=torch.mm(S_inv,force[:,None]).view(param.size()).detach() 
                    else:
                        param.grad=torch.mm(S_inv,force).view(param.size()).detach()
        
#        self.real_comp.zero_grad(); self.imag_comp.zero_grad()
#        self.SR_alg(self.real_comp,s,m_r,E_arg)
#        # Compute SR for real component
#        self.SR_alg(self.imag_comp,s,m_i,E_arg)
        
        return
        
#    def SR_alg(model, s, m, E_arg, lambduh=1, cutoff=1e-8):
#        
##        model.zero_grad()
#        N_samples=s.shape[0]
#        
#        if not hasattr(model,'autograd_hacks_hooks'):             
#            autograd_hacks.add_hooks(model)
#        outr=model(s)
#        outr.mean().backward()
#        autograd_hacks.compute_grad1(model) #computes grad per sample for all samples
#        autograd_hacks.clear_backprops(model)
#        pars=list(model.parameters())
#            
#        for param in pars:
#            with torch.no_grad():
#                if len(param.size())==2:#different mat mul rules depending on mat shape
#                    ein_str="i,ijk->ijk"
#                elif len(param.size())==1:
#                    ein_str="i,ik->ik"
#                if len(param.size())>1:
#                    if param.size(1)>param.size(0): # pytorch flips matrix pattern sometimes
#    # have to manually flip it back. (else get S=scalar for Nx1 matrix xforms-can't be right)
#                        param.grad1=param.grad1.view(param.grad1.size(0),param.size(1),param.size(0))
#                Ok=np.einsum(ein_str,m,param.grad1.numpy())
#                if len(np.shape(Ok))==2:
#                    Ok=Ok[:,:,None] 
#    # Vector bias values do not agree with original method if this is not present
#    # When present though, returns values similar in order to the other param grad values...
#                Exp_Ok=np.mean(Ok,0) # conj(mean)=mean(conj)
#    #T1=np.tensordot(np.conj(Ok_list[kk]),Ok_list[kk].T, axes=((0,2),(2,0)))/N_samples
#                T1=np.einsum("kni,imk->nm",np.conj(Ok),Ok.T)/N_samples
#            # These are methods are equivalent! Good sanity check (einsum more versitile)
#                St=2*np.real(T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)) # the S+c.c. term
#     # In NKT, cutoff of 1e-10 is used for S before inverting - inc. numerical stability?
#     # if diagS< cutoff, S(i,i)=1 and s.row=s.col=0. 
#                St[St<cutoff]=0
#                l_reg=lambduh*np.eye(St.shape[0],St.shape[1])*np.diag(St) # regulation term
#                S_inv=torch.tensor(np.linalg.pinv(St+l_reg),dtype=torch.float) # S^-1 term with reg
#                force=torch.einsum(ein_str,torch.tensor(np.real(2*E_arg*m)\
#                ,dtype=torch.float),param.grad1).mean(0) # force/DE term
#                # Compute SR 'gradient'
#                if len(force.size())==1:  # deal with .mm issues when vector Mx1
#                    param.grad=torch.mm(S_inv,force[:,None]).view(param.size()).detach() 
#                else:
#                    param.grad=torch.mm(S_inv,force).view(param.size()).detach()
#        
#        return

    def apply_grad(self, lr=0.03):
        
        params_r=list(self.real_comp.parameters()) # get the parameters
        params_i=list(self.imag_comp.parameters())    
        
        # apply the Energy gradient descent
        if len(params_r)==len(params_i):
            with torch.no_grad():
                for ii in range(len(params_r)):
                    params_r[ii] -= lr*params_r[ii].grad 
                    params_i[ii] -= lr*params_i[ii].grad
        else:
            with torch.no_grad():
                for param in params_r:
                    param -= lr*param.grad
                for param in params_i:
                    param -= lr*param.grad
        
        return

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
            
            pos=np.random.randint(self.L) # position to change
            
            alt_state = self.samples[n,:].copy() # next potential state
            
            if np.random.rand()>=0.5:
                alt_state[pos] = np.real(np.exp(1j*rot)*alt_state[pos]) # flip next random position for spin
            else:
                alt_state[pos] = np.real(np.exp(-1j*rot)*alt_state[pos]) # same chance to flip other direction
            # Will have to generalize to complex vals sometime
            
            # Probabilty of the next state divided by the current
            prob = (np.square(np.abs(self.complex_out(torch.tensor(alt_state,dtype=torch.float)))))   \
            /(np.square(np.abs(self.complex_out(torch.tensor(self.samples[n,:],dtype=torch.float)))))
            
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
                log_psi_diff=np.log(psi.complex_out(torch.tensor(s_prime,\
                dtype=torch.float)).flatten())-np.log(psi.complex_out(\
                torch.tensor(s,dtype=torch.float))).flatten()
                O_loc[:,i]+= xformed_state[:,kk]*np.exp(log_psi_diff)
            # each slice of the transformed state acts as a multiplier to 
            # its respective local spin configuration state
                
    return O_loc


def kron_matrix_gen(op_list,D,N,bc):
    ''' this function generates a Hamiltonian when it consists of a sum
 of local operators. The local operator should be input at op and the 
 lattice size of the system should be input as N. The
 op can also be entered as the kron product of the two operator 
 matrices or even three with an identity mat in-between for next 
 nearest-neighbor interactions. D is the local Hilbert space size.  '''
    
    import scipy.sparse as sp
    import numpy as np
    
    # extract/convert the operator list to a large operator
    op=op_list[0]
    for ii in range(1,len(op_list)):
        op=np.kron(op,op_list[ii])
    
    sop=sp.coo_matrix(op,dtype=np.float32) # make sparse
    
    matrix=sp.coo_matrix((D**N,D**N),dtype=np.float32) # all 0 sparse 
    
    nops=int(round(np.log(len(op))/np.log(D)) )
    #number of sites the entered op is acting on
    
    bc_term=(nops-1)
    
    for j in range(N-bc_term):
        a=sp.kron(sp.eye(D**j),sop)
        b= sp.kron(a,sp.eye(D**(N-j-nops)))
        matrix=matrix+b
    
    if bc=='periodic':
        for kk in range(nops-1):
            end_ops=op_list[-1]
            for ii in range(kk):
                end_ops=sp.kron(op_list[-ii-2],end_ops)
            
            begin_ops=op_list[0]
            for ii in range(nops-2-kk):
                begin_ops=sp.kron(begin_ops,op_list[ii+1])
            
            a=sp.kron(end_ops,sp.eye(D**(N-nops)))
            b=sp.kron(a,begin_ops)
            matrix=matrix+b
            
    return matrix








# Previously functioning SR and Grad methods (much slower for Psi vector because of loop)
    
#    def energy_gradient(self,s,E_loc,E=None): # add Pytorch optimizer) (fixed lr for now)
#        
#        if E is None:
#            E=np.mean(E_loc)
#        
#        outr = self.real_comp(s).flatten()
#        outi = self.imag_comp(s).flatten()
#        
#        E=np.conj(E)
#        E_loc=np.conj(E_loc)
#        diff=(E_loc-E)
#        mult=torch.tensor(np.real(2*diff),dtype=torch.float)
#        
#        self.real_comp.zero_grad()
#        self.imag_comp.zero_grad()
#        # should be the simpler form to apply dln(Psi)/dw_i
#        if self.form.lower()=='euler':
#            # each form has a slightly different multiplication form
#            (outr.log()*mult).mean().backward()
#            # calling this applies autograd to tensor .grad object i.e. out*mult
#            # which corresponds to dpsi_real(s)/dpars. 
#            # must include the averaging over # samples factor myself 
#            
#            # Angle
#            multiplier = 2*np.imag(-E_loc)
#            multiplier=torch.tensor(multiplier,dtype=torch.float)
#            (multiplier*outi).mean().backward()
#            
#        elif self.form.lower()=='vector':
#            if np.all(self.complex==0): 
#        # could create errors if doesn't use the updated ppsi and new s
#        # but each call of O_local redefines the .complex
#                self.complex_out(s) # define self.complex
#            
#            N_samples=s.size(0)
#            
#            p_r=list(self.real_comp.parameters())
#            p_i=list(self.imag_comp.parameters())   
#            
#            grad_list_r=copy.deepcopy(p_r)
#            grad_list_i=copy.deepcopy(p_i)
#            with torch.no_grad():
#                for param in grad_list_r:
#                    param.copy_(torch.zeros_like(param))
#                    param.requires_grad=False
#                for param in grad_list_i:
#                    param.copy_(torch.zeros_like(param))
#                    param.requires_grad=False
#
#            # what we calculated the gradients should be
#            for n in range(N_samples):
#                
#                self.real_comp.zero_grad() # important so derivatives aren't summed
#                self.imag_comp.zero_grad()    
#                outr[n].backward(retain_graph=True) # retain so that buffers aren't cleared 
#                                                    # and it can be applied again
#                outi[n].backward(retain_graph=True)
#                                                    
#                with torch.no_grad():        
#                    m= ((E_loc[n]-E)/self.complex[n]) 
#                    # [E_l*-E]/Psi according to derivative
#                    m_r=torch.tensor(2*np.real(m) ,dtype=torch.float)
#                    m_i=torch.tensor(2*np.real(1j*m) ,dtype=torch.float)
#                    
#                for kk in range(len(p_r)):
#                    with torch.no_grad():
#                        grad_list_r[kk]+=(p_r[kk].grad)*(m_r/N_samples)
#                for kk in range(len(p_r)):
#                    with torch.no_grad():
#                        grad_list_i[kk]+=(p_i[kk].grad)*(m_i/N_samples)
#            
#            # manually do the mean
#            for kk in range(len(p_r)):
#                p_r[kk].grad=grad_list_r[kk]
#            for kk in range(len(p_i)):
#                p_i[kk].grad=grad_list_i[kk]            
#                        
#        # for testing purposes
##        pr1_grad=params[0].grad
##        pi1_grad=params[0].grad
#            
#        return # pr1_grad, pi1_grad

#    def SR(self,s,E_loc, lambduh=1):
#               
#        E0=np.real(np.mean(E_loc))
#        N_samples=s.size(0)
#        
#        outr=self.real_comp(s)
#        outi=self.imag_comp(s)
#        
#        if self.form=='vector':
#            if np.all(self.complex==0):
#                self.complex_out(s)
#        
#        p_r=list(self.real_comp.parameters())
#        p_i=list(self.imag_comp.parameters())
#        
#        grad_list_i=copy.deepcopy(p_i)
#        with torch.no_grad():
#        
#            for param in grad_list_i:
#                param.copy_(torch.zeros_like(param))
#                param.requires_grad=False
#        # have to make a copy to record the gradient variable Ok and the force DE
#        Ok_list_r=[]
#        Ok_list_i=[]
#        with torch.no_grad():
#            grad_list_r=copy.deepcopy(p_r)
#            for ii in range(len(p_r)):
#                grad_list_r[ii].copy_(torch.zeros_like(p_r[ii]))
#                grad_list_r[ii].requires_grad=False
#                if len(p_r[ii].size())==1:
#                    sz1,sz2=p_r[ii].size(0),1    
#                else:
#                    sz1,sz2=p_r[ii].size()
#                Ok_list_r.append(np.zeros([N_samples,sz1,sz2],dtype=complex))
#                
#            grad_list_i=copy.deepcopy(p_i)
#            for ii in range(len(p_i)):
#                grad_list_i[ii].copy_(torch.zeros_like(p_i[ii]))
#                grad_list_i[ii].requires_grad=False
#                if len(p_i[ii].size())==1:
#                    sz1,sz2=p_i[ii].size(0),1    
#                else:
#                    sz1,sz2=p_i[ii].size()
#                Ok_list_i.append(np.zeros([N_samples,sz1,sz2],dtype=complex))
#                
#        # what we calculated the gradients should be
#        for n in range(N_samples):
#            
#            self.real_comp.zero_grad()
#            self.imag_comp.zero_grad()
#        
#            outr[n].backward(retain_graph=True) # retain so that buffers aren't cleared 
#            outi[n].backward(retain_graph=True)     # and it can be applied again
#            
#            # get the multipliers (Ok=dpsi*m) and the energy gradients for each term
#            if self.form=='vector':
#                m_r=(1/self.complex[n])
#                m_i=1j*m_r
#            else:
#                m_r=1/outr[n].detach().numpy()
#                m_i=1j
#            
#            # term for the force
#            E_arg=(np.conj(E_loc[n])-np.conj(E0))
#                  
#            for kk in range(len(p_r)):
#                with torch.no_grad():
#                    grad_list_r[kk]+=(p_r[kk].grad)*torch.tensor(\
#                    (2*np.real(E_arg*m_r)/N_samples),dtype=torch.float)
#                    Ok=p_r[kk].grad.numpy()*m_r
#                    # to deal with 1-dim params
#                    if len(np.shape(Ok))==1:
#                        Ok=Ok[:,None]
#        #            E_Ok=np.mean(Ok,1)[:,None]
#        #            S=2*np.real(np.matmul(np.conj(Ok),Ok.T)-\
#        #                        np.matmul(np.conj(E_Ok),E_Ok.T))
#                    Ok_list_r[kk][n]=Ok
#        
#            for kk in range(len(p_i)):
#                with torch.no_grad():
#                    grad_list_i[kk]+=(p_i[kk].grad)*torch.tensor(\
#                    (2*np.real(E_arg*m_i)/N_samples),dtype=torch.float)
#                    Ok=p_i[kk].grad.numpy()*m_i
#                    if len(np.shape(Ok))==1:
#                        Ok=Ok[:,None]
#                    Ok_list_i[kk][n]=Ok
#        # unfortunately, must record Ok for each sample so an expectation <Ok> can be taken
#        # This could be a memory/speed issue, but I don't see an obvious route around it
#                    
#        S_list_r=[]
#        for kk in range(len(Ok_list_r)):
#            Exp_Ok=np.mean(Ok_list_r[kk],0)  # conj(mean)=mean(conj)
#        #    T1=np.tensordot(np.conj(Ok_list[kk]),Ok_list[kk].T, axes=((0,2),(2,0)))/N_samples
#            T1=np.einsum('kni,imk->nm',np.conj(Ok_list_r[kk]),Ok_list_r[kk].T)/N_samples
#            # These are methods are equivalent! Good sanity check
#            St=2*np.real(T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)) # the S+c.c. term
#            l_reg=lambduh*np.eye(St.shape[0],St.shape[1])*np.diag(St) # regulation term
#            S_list_r.append(St+l_reg) 
#        
#        S_list_i=[]
#        for kk in range(len(Ok_list_i)):
#            Exp_Ok=np.mean(Ok_list_i[kk],0) 
#            T1=np.einsum('kni,imk->nm',np.conj(Ok_list_i[kk]),Ok_list_i[kk].T)/N_samples
#            St=2*np.real(T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)) # the S+c.c. term
#            l_reg=lambduh*np.eye(St.shape[0],St.shape[1])*np.diag(St) # regulation term
#            S_list_i.append(St+l_reg) 
#        
#        for kk in range(len(p_r)):
#            S_inv=torch.tensor(np.linalg.pinv(S_list_r[kk]),dtype=torch.float) # have to inverse S
#            if len(grad_list_r[kk].size())==1: # deal with .mm issues when vector Mx1        
#                p_r[kk].grad=(torch.mm(S_inv,grad_list_r[kk][:,None]))\
#                .view(p_r[kk].size()).detach()
#            else:
#                p_r[kk].grad=torch.mm(S_inv,grad_list_r[kk])
#        
#        for kk in range(len(p_i)):
#            S_inv=torch.tensor(np.linalg.pinv(S_list_i[kk]),dtype=torch.float) # have to inverse S
#            if len(grad_list_i[kk].size())==1: # deal with .mm issues when vector Mx1        
#                p_i[kk].grad=(torch.mm(S_inv,grad_list_i[kk][:,None]))\
#                .view(p_i[kk].size()).detach()
#            else:
#                p_i[kk].grad=torch.mm(S_inv,grad_list_i[kk])
#            
#        return 