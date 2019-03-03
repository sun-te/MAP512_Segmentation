#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:09:29 2019

@author: tete
"""

import numpy as np
import os
from PIL import Image
os.chdir("/home/tete/Study/MAP512/Projet_MAP512")
from Scripts import Random_walker,EclipseGenerator

data_dir = '/home/tete/Study/MAP512/Projet_MAP512/data/'
#%%
img_name=data_dir+"noise/Noise_13.png"
im=Image.open(img_name)
L=im.convert('L')
mat_noise=np.array(L)
mat_noise=mat_noise/np.max(mat_noise)

img_name=data_dir+"pure/Pure_13.png"
im=Image.open(img_name)
L=im.convert('L')

mat_pure=np.array(L)

#normalisation nessessary!!!!!!
mat_pure=mat_pure/np.max(mat_pure)


e=EclipseGenerator.Eclipse(1,1,N=256)
e.img=mat_pure
e.Plot()
#%%
seeds=[[int(len(mat_pure)/2),int(len(mat_pure)/2)],[250,250],[1,134]]
labels=[0,1,1]
beta=500
[mask,proba]=Random_walker.random_walker(mat_pure,seeds,labels,beta)
e.img=mask
e.Plot()







#%%
N=256
e=EclipseGenerator.Eclipse(0.5,0.8,rotation_degree=39,N=N)
e.Generate(back_ground="white")
seeds=[[int(N/2),int(N/2)],[N,N]]
labels=[0,1]
beta=10
e.Plot()
#%%
[mask,proba]=Random_walker.random_walker(e.img,seeds,labels,beta)

e=EclipseGenerator.Eclipse(1,1,N=N)
e.img=mask
e.Plot()










#%%
img_name=data_dir+"pure/axial_CT_slice.bmp"
im=Image.open(img_name)
L=im.convert('L')
mat_img=np.array(L)
mat_img=mat_noise/np.max(mat_img)
e=EclipseGenerator.Eclipse(1,1,N=256)
e.img=mat_img
e.Plot()
#%%
beta=100
seeds=[[130,150],[200,200],[250,250]]
labels=[0,1,1]

[mask,proba]=Random_walker.random_walker(mat_img,seeds,labels,beta)

e=EclipseGenerator.Eclipse(1,1,N=256)
e.img=mask
e.Plot()

e.img=mat_img+mask
e.Plot()


