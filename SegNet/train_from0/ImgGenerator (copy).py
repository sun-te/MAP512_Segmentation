#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
    test
    Created on Tue Nov 20 14:50:21 2018
    @author: TeTe & Zeal
'''

import numpy as np
import os
os.chdir("/home/tete/Study/MAP512/Projet_MAP512")
from Scripts import EclipseGenerator

#from importlib import reload
#reload(EclipseGenerator)
#%%
##Test          

pure_path="/home/tete/Study/MAP512/Projet_MAP512/data/pure/"
e=EclipseGenerator.Eclipse(0.8,0.4,center=[0.0,0.2],rotation_degree=25,N=501)
e.S_noise(num=50,scale=0.05)
e.Plot()
e.G_Noise(scale=0.3)
e.Plot()
#e.Save_plot(pixel=200,pure_path+"test")
#%%
#save some shape noise and gaussian noise
pure_path="/home/tete/Study/MAP512/Projet_MAP512/data/pure/"
noise_path="/home/tete/Study/MAP512/Projet_MAP512/data/noise/"
M=20
for i in range(M):
    a,b=np.random.uniform(low=0.1,high=1.0, size=2)
    deg=np.random.uniform(0.0,180.0)
    
    e=EclipseGenerator.Eclipse(a,b,center=[0.0,0.0],rotation_degree=deg,N=501)
    
    #Generate & Save label image    
    num=np.random.randint(low=0,high=51)           #number of shape noise
    scale=np.random.uniform(low=0.0, high=0.1)     #shape noise length scale
    e.S_noise(num=num,scale=scale)
    name="Pure_"+str(i)
    e.Save_plot(pixel=256,save_name=pure_path+name)
    
    #Generate & Save noisy image
    e.G_Noise(scale=np.random.uniform())           #variance of the gaussian noise
    name="Noise_"+str(i)
    e.Save_plot(pixel=256,save_name=noise_path+name)

#%%
Mp=10
for i in range(M,M+Mp):
    a,b=np.random.uniform(low=0.1,high=1.0, size=2)
    deg=np.random.uniform(0.0,180.0)
    
    e=EclipseGenerator.Eclipse(a,b,center=[0.0,0.0],rotation_degree=deg,N=501)
    e.Generate(back_ground= "white"if(np.random.uniform()<0.7) else "black")
    
    name="Pure_"+str(i)
    e.Save_plot(pixel=256,save_name=pure_path+name)
    
    e.G_Noise(scale=np.random.uniform())           #variance of the gaussian noise
    name="Noise_"+str(i)
    e.Save_plot(pixel=256,save_name=noise_path+name)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#%%  
    
    
    
    
    