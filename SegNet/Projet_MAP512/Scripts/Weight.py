#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 00:00:41 2019

@author: tete

Attention the X and Y here defined is different from the ususally one!!!!

X=2, Y=3

[[1,1],
 [1,1],
 [1,1]]

"""

import numpy as np
from scipy import sparse
#%%
'''
0 for 4 edges
1 for 8 edges
'''

def lattice(X,Y,connect=0):
    if(X*Y==1):
        points=np.array([1,1])
        edges=np.array([])
        return points,edges
    
    N=X*Y
    x=np.zeros(N)
    y=np.zeros(N)
    
    for i in range(X):
        for j in range(Y):
            x[i*Y+j]=i
            y[j*X+i]=j
    points=np.append(x,y)
    
    
    #edges
    pos=np.array([[i for i in range(1,N+1)]]).reshape(Y,X)
    edges=[];

    for i in range(Y):
        for j in range(X):
            #left
            if(i-1>=0):
                try:
                    
                    a,b=min(pos[i,j],pos[i-1,j]),max(pos[i,j],pos[i-1,j])
                    edges.append((a,b))
                except:
                    s="list out of bound"
            #right
            try:
                a,b=min(pos[i,j],pos[i+1,j]),max(pos[i,j],pos[i+1,j])
                edges.append((a,b))
            except:
                s="list out of bound"
            #up
            if(j-1>=0):
                try:
                    a,b=min(pos[i,j],pos[i,j-1]),max(pos[i,j],pos[i,j-1])
                    edges.append((a,b))
                except:
                    s="list out of bound"
            #down
            try:
                a,b=min(pos[i,j],pos[i,j+1]),max(pos[i,j],pos[i,j+1])
                edges.append((a,b))
            except:
                s="list out of bound"
    #8-neighbors
    if(connect==1):
        for i in range(Y):
            for j in range(X):
                if(i-1>=0):
                    try:
                        a,b=min(pos[i,j],pos[i-1,j-1]),max(pos[i,j],pos[i-1,j-1])
                        edges.append((a,b))
                    except:
                        s="list out of bound"
                    try:
                        a,b=min(pos[i,j],pos[i-1,j+1]),max(pos[i,j],pos[i-1,j+1])
                        edges.append((a,b))
                    except:
                        s="list out of bound"
                #right
                try:
                    a,b=min(pos[i,j],pos[i+1,j+1]),max(pos[i,j],pos[i+1,j+1])
                    edges.append((a,b))
                except:
                    s="list out of bound"
                
                if(j-1>=0):
                    try:
                        a,b=min(pos[i,j],pos[i+1,j-1]),max(pos[i,j],pos[i+1,j-1])
                        edges.append((a,b))
                    except:
                        s="list out of bound"       
    edges=list(set(edges))
    return points,edges
#%%
def point2pos(p,X):
    return  int((p-1)/X),(p-1)%X
    
#%%
'''
beta: an outer parameter to choose
'''
def makeweights(edges,img,beta):
    Y,X=img.shape[0],img.shape[1]
    #numerical stablilty
    eps=1e-5
    
    M=len(edges)
    weights=np.zeros([M,1])
    for i in range(M):
        p1,p2=edges[i]
        #print(edges[i])
        contrast=img[point2pos(p1,X)]-img[point2pos(p2,X)]
        #norm2
        weights[i]=contrast**2
        
    Max,Min=max(weights),min(weights)
    if(Max==Min):
        return np.ones([M,1])
    else:#normalisation
        weights-=Min
        weights/=(Max-Min)
    return np.exp(-beta *weights+eps)
        
        
def degree(label, edges,weights):
    deg=0.0
    M=len(edges)
    for i in range(M):
        for j in range(2):
            if(edges[i][j]==label):
                deg+=weights[i][0]
    return deg

def degreeGraph(img,edges,weights):
    Y,X=img.shape[0],img.shape[1]
    degs=np.zeros([Y,X])
    for i in range(Y):
        for j in range(X):
            label=i*X+j+1
            degs[i,j]=degree(label, edges,weights)
    return degs

#laplacian
def Adjacency(edges,img,weights):
    N=img.shape[0]*img.shape[1]
    A=sparse.lil_matrix((N, N))
    #A=np.zeros([N,N])
    #A=A.tolil()
    for e in range(len(edges)):
        edge=edges[e]
        A[int(edge[0]-1),int(edge[1]-1)]=weights[e]
    A=A+A.T
    return A
    
def Laplacian(edges,img,weights):
    A=Adjacency(edges,img,weights)
    diag=A.sum(axis=1).reshape(1,-1)
    diag=diag.tolist()[0]
    return sparse.diags(diag)-A


#%%
#points, edges=lattice(2,3)
#img=np.array([[1,1],[1,10],[1,1]])
#weights=makeweights(edges,img,10)
#degreeMatrix=degreeGraph(img,edges,weights)
#L=Laplacian(edges,img, weights)
    









