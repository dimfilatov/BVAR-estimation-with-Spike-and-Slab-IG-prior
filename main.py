# -*- coding: utf-8 -*-


import os
from pathlib import Path
import numpy as np 
import scipy.linalg as spla 
import pandas as pd
import matplotlib.pyplot as plt
import math

#set the working directory to source file location
mypath = Path().absolute()
os.chdir(mypath)

# Define functions
def datasim(n,p,prob_incl):
    #Data Simulation
    s = (p,p)
    beta=np.zeros(s) #inizialize to zeros all coef.
    #index=np.around(np.random.rand(nonzeros)*(p-1)) #randomly select the nonzeros
    #index=[int(i) for i in index] #convert to integer

    stationary=0
    while stationary==0:
        for i in range(0,p):
            for j in range(0,p):
                if np.random.uniform(0,1)<prob_incl:
                    beta[i,j]=np.random.normal(0.15,0.05,1) #set some coef to "non-zero"
                else: 
                    beta[i,j]=0
        if max(abs(spla.eigh(beta,eigvals_only=True)))<1:
            stationary=1  
    A =  np.identity(p)       
    for i in range(0,p):
        for j in range(0,i):
            A[i,j]=np.random.normal(0,0.5,1)
    invA = np.linalg.inv(A)
    s = (n,p)
    Y1=np.zeros(s)
    for i in range(1,n):
        Y1[i,:]=np.matmul(Y1[i-1,:],beta.T) + np.matmul(invA,np.random.normal(0,1,p))
    burn=50
    Y=Y1[2+burn:n,:]
    X=Y1[1+burn:n-1,:]

    return(Y,X)
    
def sampler_b(Y,X,z,q,sigma2):
    #the full conditional of b is a multivariate normal for z=1 and a degenerate in zero for z=0.

    #STEP 0: select the non zero coef.
    tau=np.sum(z) #how many
    tildaX=X[:,np.nonzero(z)[0]] #which ones
    
    #STEP 1: compute the var-cov matrix (sigma2*tildaW^(-1))
    tildaW=np.matmul(tildaX.T,tildaX)+np.identity(tau)
    covb=sigma2*np.linalg.inv(tildaW)
    
    #STEP 2: compute the mean vector (tildaW^(-1)*(tildaX*Y))
    betahat=np.matmul(np.linalg.inv(tildaW),np.matmul(tildaX.T,Y))
    
    #STEP 3: sample those b's which differ from 0
    Chol=spla.cholesky(covb) #Cholesky of the var-cov matrix
    bnonzero=np.random.randn(tau) #uncorrelated zero mean
    bnonzero=bnonzero.dot(Chol)+betahat #simulated non-zeros coef. 
    #alternative: np.random.multivariate_normal(betahat, covb)
    
    #STEP 4: "insert" the zero coeff:
    b=np.zeros(X.shape[1])
    b[np.nonzero(z)[0]]=bnonzero
    
    return b

def sampler_q(z):
    #the full conditional of q is a beta(tau+tau/2+1,K-tau+1)
     
    #STEP 0:compute the parameters
    tau=np.sum(z)
    param1=tau+tau/2+1
    param2=z.size-tau+1
    
    #STEP 1: sample q
    q=np.random.beta(param1,param2)
    
    return q

def sampler_sigma2(Y,X,z,q):
    #not using b 'cause we are block sampling b and sigma2
    #the conditional distribution of sigma2 is InverseGamma
    
    #STEP 0:compute the parameters
    
    #num of obs (the previous n in the simulation of data): 
    T=Y.size #(or T=Y.shape[0])
    
    #number of nonzeros coef.
    tau=np.sum(z)
    
    #tilda X:
    tildaX=X[:,np.nonzero(z)[0]] 
    
    #tilda W:
    tildaW=np.matmul(tildaX.T,tildaX)+np.identity(tau)
    
    #beta tilda hat:(tildaW^(-1)*(tildaX*Y))
    betahat=np.matmul(np.linalg.inv(tildaW),np.matmul(tildaX.T,Y))
    
    #final param for the InverseGamma
    paramA=T/2
    paramB=np.matmul(Y.T,Y)-np.matmul(np.matmul(betahat.T,tildaW),betahat)
    paramB=paramB/2
    
    #STEP 1: sample sigma2
    sigmatemp=np.random.gamma(paramA,paramB)
    sigma2=1/sigmatemp
    
    return sigma2

def sampler_zi(Y,X,z,q,i):
    
    #STEP 0:compute the needed dimensions
    
    #num of obs (the previous n in the simulation of data): 
    T=Y.shape[0] #(or T=Y.shape[0])
    
    #tot number of regressors (the previous p in the simulation of data)
    K=X.shape[1]
       
    #STEP 2:p(z_i=1|z-i,theta) up to a constant (i.e. the joint cond. on theta)
    
    #tau (when zi=1)
    z[i]=1
    tau=np.sum(z) #equal to the one before plus 1
    
    #tilda W:
    tildaX=X[:,np.nonzero(z)[0]] 
    tildaW1=np.matmul(tildaX.T,tildaX)+np.identity(tau)
    
    #beta tilda hat:(tildaW^(-1)*(tildaX*Y))
    betahat=np.matmul(np.linalg.inv(tildaW1),np.matmul(tildaX.T,Y))
      
    #finally the joint (i.e. conditional of z_i=0 up to a constant)
    #Eliminate q part since it goes away when we compute parameter p further
    #Excluding q part is crucial because otherwise we may face problem of extreme values
    
    F1=np.matmul(Y.T,Y)-np.matmul(np.matmul(betahat.T,tildaW1),betahat)

    
    #STEP 1:p(z_i=0|z-i,theta) up to a constant (i.e. the joint cond. on theta)
    
    #tau (when zi=0)
    z[i]=0
    tau=np.sum(z)
    
    #tilda W:
    tildaX=X[:,np.nonzero(z)[0]] 
    tildaW0=np.matmul(tildaX.T,tildaX)+np.identity(tau)
    
    #beta tilda hat:(tildaW^(-1)*(tildaX*Y))
    betahat=np.matmul(np.linalg.inv(tildaW0),np.matmul(tildaX.T,Y))
    
    if tau==0:
        F0=np.matmul(Y.T,Y)
        p=1/(1+np.power((1/np.linalg.det(tildaW1)),(-1/2))*(1-q)/q*np.power((F0/F1),(-T/2)))
    else:
        F0=np.matmul(Y.T,Y)-np.matmul(np.matmul(betahat.T,tildaW0),betahat)
        p=1/(1+np.power((np.linalg.det(tildaW0)/np.linalg.det(tildaW1)),(-1/2))*(1-q)/q*np.power((F0/F1),(-T/2)))
    #finally the joint (i.e. conditional of z_i=0 up to a constant)      
   
    
    #STEP 4: sample zi
    zi=np.random.binomial(1,p)
    
    return zi

def sampler_invA(resid):
    X_adj = resid[:,0]
    for i in range(1,p):
        y_adj=resid[:,i]
        ZZ=np.matmul(X_adj,X_adj.T)
        Zz=np.matmul(X_adj,y_adj.T)
        if i==1:
            Valpha_post=1/ZZ
            alpha_post=Valpha_post*Zz
            alphadraw=alpha_post + math.sqrt(Valpha_post)*np.random.normal(0,1,1)
        else:
            Valpha_post=np.linalg.inv(ZZ)
            alpha_post=np.matmul(Valpha_post,Zz)
            alphadraw=alpha_post + np.matmul(np.linalg.cholesky(Valpha_post),np.random.normal(0,1,i)) #transpose(chol(V)))
        A[i,0:i]=-1*alphadraw
        X_adj=np.vstack((X_adj,resid[:,i]))
    invA=np.linalg.inv(A) # use this matrix in Gibbs sampler further to apply triangulazition
    
    return invA

#1. SIMULATE DATA 
#np.random.seed(1) #for reproducibility 


n=200 #sample size
p=15 #n.of regressors 
prob_incl=0.5 #n. of "relevant" regressor (i.e.num. of non-zeros coef.)

data=datasim(n,p,prob_incl)
Y=data[0]
X=data[1]
n=X.shape[0]

#2. POSTERIOR ESTIMATE

#2.1 inizialize values 
q=np.ones(p) #doesn matter since is the first variable to be sampled
s1 = (p,p)
b=np.zeros(s1)
z=1*(b!=0)

#2.2 Gibbs sampler: 
ite=200 #n. of MCMC iteration
burn_in=50 #burn_in size 

#"prepare" matrix where to save the samples
s2=(ite,p,p)
bsample=np.zeros(s2)
s3=(ite,p)
qsample=np.zeros(s3)
zsample=np.zeros(s2)
invA=np.identity(p)
A=np.identity(p)
s4=(n,p)
resid=np.zeros(s4)

for it in range(0,ite):
    print('MCMC iteration=',it)
    stationary = 0
    while stationary==0:
        for i in range(0,p):
            zi=z[i,:]
            q[i]=sampler_q(np.asarray(zi))
            #Now apply triangulazition transformation
            Yr=Y[:,i]
            if i>0:
                for l in range(0,i-1):
                    Yr=Yr-resid[:,l] * invA[i,l] 
            for j in range(0,p):
                z[i,j]=sampler_zi(Yr,X,np.asarray(zi),q[i],j)
            sigma2=sampler_sigma2(Yr,X,np.asarray(zi),q[i])
            b[i,:]=sampler_b(Yr,X,z[i,:],q[i],sigma2) 
            qsample[it,i]=q[i]
            bsample[it,i,:]=b[i,:]
            zsample[it,i,:]=z[i,:]
        if max(abs(spla.eigh(b,eigvals_only=True)))<1:
            stationary=1
 # Now save new innovations to be subtracted from VAR in the triangulazition transformation
    resid=Y-np.matmul(X,b.T)
    invA=sampler_invA(resid)
