#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:51:18 2020

@author: alex
"""

import torch
import torch.nn as nn
import numpy as np

class QNADE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, evals ):
        super(QNADE, self).__init__()
        
        self.evals=np.array(evals) # evals the model can sample
        self.D = input_dim # the system and input size
        self.H = hidden_dim 
        self.M = len(evals)*input_dim # this is the output size, 
                                     # a probability for each eval

        self.params = nn.ParameterDict({
                # visible to hidden linear matrix x-form, imaginary comp
                'Wr' : nn.Parameter(torch.zeros([self.H,self.D])), 
                # hidden layer bias, imaginary comp
                'cr' : nn.Parameter(torch.zeros([1, self.H])),
                # hidden to output layer matrix x-form, imaginary comp
                'Vr' : nn.Parameter(torch.zeros([self.M,self.H])),
                # output bias vector, imaginary comp
                'br' : nn.Parameter(torch.zeros([self.M])),
                
                # visible to hidden linear matrix x-form, imaginary comp
                'Wi' : nn.Parameter(torch.zeros([self.H,self.D])), 
                # hidden layer bias, imaginary comp
                'ci' : nn.Parameter(torch.zeros([1, self.H])),
                # hidden to output layer matrix x-form, imaginary comp
                'Vi' : nn.Parameter(torch.zeros([self.M,self.H])),
                # output bias vector, imaginary comp
                'bi' : nn.Parameter(torch.zeros([self.M])),})
    
    # TODO add some customizability to the ANNs that transform the samples to vi
    
        nn.init.xavier_normal_(self.params['Vr'])
        nn.init.xavier_normal_(self.params['Wr'])
        nn.init.xavier_normal_(self.params['Vi'])
        nn.init.xavier_normal_(self.params['Wi'])

    def forward(self, N_samples=None, x=None): 
                
        if N_samples is None and x is None: 
            raise ValueError('Must enter spin states for Psi calculation or the number of samples to be generated')
        if N_samples is None and x is not None: N_samples, sample = x.shape[0], False
        if N_samples is not None and x is None: sample = True
        
        # a_0, d=0 is set to c, and updated on each run. expanded to be sample size
        a_dr = self.params['cr'].expand(N_samples, -1) 
        a_di = self.params['ci'].expand(N_samples, -1)
        
        # the full Psi is a product of the conditionals, making a running product easy
        PPSI=np.ones([N_samples],dtype=np.complex128) # if multiplying
        #PPSI=np.zeros([N_samples],dtype=np.complex128)  # if adding logs
        
        # number of outputs we must get for the output layer
        nevals = len(self.evals)
        
        for d in range(self.D):
            # This is the hidden layer activation
            h_dr = torch.sigmoid(a_dr)
            h_di = torch.sigmoid(a_di)
            
            # Calculate the visible layer output (v_i)
            vi_dr =h_dr.mm(self.params["Vr"][nevals*d:nevals*(d+1), :].t())\
                    +self.params["br"][nevals*d:nevals*(d+1)]
                    
            vi_di =h_di.mm(self.params["Vi"][nevals*d:nevals*(d+1), :].t())\
                    +self.params["bi"][nevals*d:nevals*(d+1)]
                    
            # might need to end the forward here, can't autocalc derivs when complex #'s involved
            # alternatively, I could accumulate the gradients in the forward/sampling pass.
#            vi_dr.backward() # needs to be scalar or need to use autograd_hacks
#            vi_di.backward() 
            
            # TODO create a variable to accumulate the gradient of vi (can be used for full grad)
            
            # The Quantum-NADE deviates from a NADE in having a real and imag comp
            # Here we can use both vi to generate a complex vi that is the 
            # basis of our calculations and sampling
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
                
                xd = torch.tensor(self.evals[samplepos],dtype=torch.float) # sample
                if len(xd.shape)==1:
                    xd = xd[:,None]
                if d==0:
                    samples = xd 
                else:
                    samples = torch.cat((samples,xd),dim=1) 
                # End sampling routine
            
            else:
                xd = x[:,d:d+1]
                if d==0: samples=xd # just checking the iterations
                else: samples = torch.cat((samples,xd),dim=1) 
                
                # find the s_i for psi(s_i), which is to be accumulated for PPSI
                samplepos = (xd==self.evals[1]).int().numpy().squeeze()
                # TODO this definitely won't work for non-binary evals 
        
            # NADE update rule, uses previously sampled x_d
            a_dr = a_dr + xd.mm(self.params['Wr'][:,d:(d+1)].t())+self.params['cr']
            a_di = a_di + xd.mm(self.params['Wi'][:,d:(d+1)].t())+self.params['ci']

            # Multiplicitavely accumulate PPSI based on which sample (s) was sampled
            PPSI=PPSI*psi[range(N_samples),samplepos]
            
            # PPSI may only make sense when inputing an x to get the wvf for...
            
        return PPSI, samples




