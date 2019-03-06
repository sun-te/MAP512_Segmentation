#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 21:36:14 2019

@author: tete
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import time


#%%

def dirichelet(laplacian,list_point, boundary):
    N=laplacian.shape[0]
    num_label=boundary.shape[1]
    
    mkd  = [i-1 for i in list_point]
    umkd =[]
    for i in range(N):
        if(not (i in mkd) ):
            umkd.append(i)
    B=(laplacian[mkd][:,umkd]).T
    Lu=laplacian[umkd][:,umkd]
    
    M=sparse.coo_matrix(Lu)
    Y=-B.dot(boundary)
    
    
    
    #To solve A * X = b
    X =np.zeros(shape=[len(umkd),num_label])
    
    
##    No zuo No Die, Why you try??? Never inverse a matrix 
#    from scipy.sparse.linalg import inv
#    time0=time.time()
#    X=inv(M).dot(Y)
#    time1=time.time()
#    print(time1-time0)
#    
    #gradient desent with adaptive step size

# =============================================================================
#     time0=time.time()
#     for l in range(num_label):
#         X[:,l]=GPO(M,Y[:,l],tol=1e-5,itermax=1e4,x0=X0[:,l])
#     time1=time.time()
#     print(time1-time0)
# =============================================================================
    
    time0=time.time()
    for l in range(num_label):
        X[:,l]=GC(M,Y[:,l],tol=1e-6,itermax=1e6,x0=np.random.uniform(size=[len(umkd),1]))
    time1=time.time()
    print("Time to compute the linear system: ",time1-time0)
    
    
    proba=np.zeros([N,num_label])
    proba[umkd,:]=X
    proba[mkd ,:]=boundary
    return proba;
    
    
    
#%%    
def norm(r):
    return np.linalg.norm(r,2)
def GPF(A,b,tol,alpha , itermax ,x0):
    err=[]
    x=x0
    r=b-A.dot(x)
    iteration=0
    
    while( iteration<itermax and tol<norm(r)):
        err.append(norm(r))
        x=x+alpha*r
        r=b-A.dot(x)
        iteration+=1
    plt.plot(err)
    return x,iteration,err


def GPO(A,b,tol,itermax,x0):
    x=x0
    err=[]
    r=b-A.dot(x)
    iteration=0
    while(iteration<itermax and  tol<norm(r)):
        err.append(norm(r))
        alpha=np.dot(r.T,r)/np.dot(A.dot(r).T,r)
        x=x+alpha*r
        r=b-A.dot(x)
        iteration+=1
    print("Gradient with optimized time step")
    plt.plot(err)
    plt.plot()
    return x

def GC(A,b,tol,itermax,x0):
    
    err=[]
    b=b.reshape([len(b),1])

    r_before=b-A.dot(x0)
    p=r_before
    
    x=x0
    iteration=0
    
    while(norm(r_before)>tol and iteration<itermax):
        err.append(norm(r_before))
        
        alpha=norm(r_before)**2/np.dot(A.dot(p).T,p).item()
        x=x+alpha *p

        
        r_after=r_before-alpha *A.dot(p)

        
        beta=norm(r_after)**2/norm(r_before)**2
        
        p=r_after+beta*p
        
        iteration+=1100
        r_before=r_after
    print("Gradient conjugué")
    plt.plot(err)
    plt.show()
    
    return x.T[0]