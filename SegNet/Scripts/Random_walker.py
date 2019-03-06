#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 10:50:39 2019

@author: tete
"""

import numpy as np

from Scripts import Weight,Dirichelet
#%%
"""
seeds: position of the seeds.
example:
    [[1,3],[34,14],[1,98]];
labels: labels of the seeds.
example:
    [1,3,1]
"""

def random_walker(img, seeds, labels, beta):
# =============================================================================
#     #test use only
#     beta=10
#    img=e.img
#    seeds=[[1,1],[3,3]]
#     img=np.random.uniform(low=0,high=255, size=[256,256])
#     seeds=[[44,2],[32,19]]
# =============================================================================
#    labels=[0,1]
    num_label=len(set(labels))
    assert(num_label<=len(seeds))
    assert(len(labels)==len(seeds))
    
    Y,X=img.shape[:]
    
    #4 neighbor model
    points,edges=Weight.lattice(X,Y,connect=0)
    weights=Weight.makeweights(edges,img,beta)
    
    lap=Weight.Laplacian(edges,img,weights)
    
    marked_point=np.zeros([Y,X])
    list_point=[]
    for s in seeds:
        assert(s[0]<=X and s[0]>0 and s[1]<=Y and s[1]>0), "Position out of the image"
        marked_point[s[1]-1,s[0]-1]=1
        list_point.append((s[0]-1)+(s[1]-1)*X+1)
    
    
    marked_point=marked_point.reshape(-1,X*Y)
    
    
    boundary=np.zeros([len(seeds),len(set(labels))])
    for s in range(len(seeds)):
        for l in range(len(labels)):
            if labels[s]==l:
                boundary[s,l]=1         
    proba=Dirichelet.dirichelet(lap,list_point,boundary)
    
    mask=np.argmax(proba,axis=1)
    
    mask=mask.reshape(Y,X)
    proba=proba.reshape([Y,X,num_label])
    
    return [mask,proba]

    
    
