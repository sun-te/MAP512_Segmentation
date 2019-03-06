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
root            ="/home/tete/SegNet-tensorflow/Seg/"
#file_name       ="/Seg/"
trainannot_path =root +"trainannot/"
train_path      =root +"train/"
test_path       =root +"test/"
testannot_path  =root +"testannot/"
val_path  =root +"val/"
valannot_path  =root +"valannot/"
#%%
def Name(i):
    name= str(i)
    l=len(name)
    for i in range(4-l):
        name='0'+name
    return name
    
#%%
'''Generate training data'''
M=20
text_train=''

for i in range(M):
    a,b=np.random.uniform(low=0.1,high=1.0, size=2)
    deg=np.random.uniform(0.0,180.0)
    
    e=EclipseGenerator.Eclipse(a,b,center=[0.0,0.0],rotation_degree=deg,N=501)
    
    #Generate & Save label image    
    num=np.random.randint(low=0,high=51)           #number of shape noise
    scale=np.random.uniform(low=0.0, high=0.1)     #shape noise length scale
    e.S_noise(num=num,scale=scale,back_ground="white")#if(np.random.uniform()<0.7) else "black")
    name=Name(i)
    
    e.Save_plot(pixel=128,save_name=trainannot_path+name)
    
    e.G_Noise(scale=np.random.uniform(0,0.3))
    e.Save_plot(pixel=128,save_name=train_path+name)
    
    #write in the log file
    text_train+="/Seg/train/"+name+'.png '+"/Seg/trainannot/"+name+'.png\n'

with open(root+"train.txt","w") as f:
    f.write(text_train)

#%%
'''Generate test data'''
M=10
text_test =''
for i in range(M):
    a,b=np.random.uniform(low=0.1,high=1.0, size=2)
    deg=np.random.uniform(0.0,180.0)
    
    e=EclipseGenerator.Eclipse(a,b,center=[0.0,0.0],rotation_degree=deg,N=501)
    
    #Generate & Save label image    
    num=np.random.randint(low=0,high=51)           #number of shape noise
    scale=np.random.uniform(low=0.0, high=0.1)     #shape noise length scale
    e.S_noise(num=num,scale=scale,back_ground="white")#if(np.random.uniform()<0.7) else "black")
    name=Name(i)
    e.Save_plot(pixel=128,save_name=testannot_path+name)
    
    
    e.G_Noise(scale=np.random.uniform(0,0.3))
    e.Save_plot(pixel=128,save_name=test_path+name)
    #write in the log file
    text_test+="/Seg/test/"+name+'.png '+"/Seg/testannot/"+name+'.png\n'


    #Generate & Save noisy image
    #e.G_Noise(scale=np.random.uniform())           #variance of the gaussian noise
    #name="Noise_"+str(i)
    #e.Save_plot(pixel=256,save_name=noise_path+name)
with open(root+"test.txt","w") as f:
    f.write(text_test)
#with open(root+"train.txt","w") as f:
#    f.write(text_train)
#%%
'''Generate val data'''
M=101
val_test =''
for i in range(M):
    a,b=np.random.uniform(low=0.1,high=1.0, size=2)
    deg=np.random.uniform(0.0,180.0)
    
    e=EclipseGenerator.Eclipse(a,b,center=[0.0,0.0],rotation_degree=deg,N=501)
    
    #Generate & Save label image    
    num=np.random.randint(low=0,high=51)           #number of shape noise
    scale=np.random.uniform(low=0.0, high=0.1)     #shape noise length scale
    e.S_noise(num=num,scale=scale,back_ground="white")#if(np.random.uniform()<0.7) else "black")
    name=Name(i)
    e.Save_plot(pixel=64,save_name=val_path+name)
    e.Save_plot(pixel=64,save_name=valannot_path+name)
    
    #write in the log file
    val_test+=val_path+name+'.png '+valannot_path+name+'.png\n'


    #Generate & Save noisy image
    #e.G_Noise(scale=np.random.uniform())           #variance of the gaussian noise
    #name="Noise_"+str(i)
    #e.Save_plot(pixel=256,save_name=noise_path+name)
with open(root+"val.txt","w") as f:
    f.write(val_test)
#with open(root+"train.txt","w") as f:
#    f.write(text_train)
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
    
import numpy as np
import os
from PIL import Image
os.chdir("/home/tete/Study/MAP512/Projet_MAP512")
from Scripts import Random_walker,EclipseGenerator
data_dir = '/home/tete/SegNet/CamVid/testannot'    
img_name=data_dir+"/0001TP_008550.png"
im=Image.open(img_name)
L=im.convert('L')
mat_noise=np.array(L)
mat_noise=mat_noise/np.max(mat_noise)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #%%
    
    
    
    
    