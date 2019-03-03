#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 00:59:14 2019

@author: tete
"""

import numpy as np
from matplotlib import pyplot as plt
#%%
#To transform onto the axis- x and y
def Fit_axis(matrix):
    
    tmp = matrix.T
    return tmp[::-1]
def Reverse_axis(matrix):
    tmp=matrix.T
    return tmp[::-1]

class Eclipse():
    def __init__(self,a,b,center=[0.,0.],rotation_degree=0,N=101):
        self.a=a
        self.b=b
        self.center=center
        theta=rotation_degree*1./180*np.pi
        self.rotation=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        self.img=np.zeros([N,N])
        
    def Generate(self,back_ground="white"):
        N=len(self.img)
        temp=np.zeros([N,N])
        mid=int((N-1)/2)
        a_pixel=self.a*mid
        b_pixel=self.b*mid
        if(back_ground=="white"):
            fond=-1
        else:
            fond=1
            
        for i in range(N):
            for j in range(N):
                new_axis=self.rotation.dot(
                    np.array([[i-(self.center[0]+1)*mid],[j-(self.center[1]+1)*mid]]))
                
                x,y=new_axis[0][0],new_axis[1][0]
                
                                                      
                if((x)**2/a_pixel**2 +(y)**2/b_pixel**2<=1):
                    color=-1
                    temp[i,j]=(1-fond*color)/2
                else:
                    color=1
                    temp[i,j]=(1-fond*color)/2
        self.img=Fit_axis(temp)
    '''
    For gaussian noise on every pixel
    '''
    def G_Noise(self,scale=0.1):
        N=len(self.img)
        noise=np.random.normal(size=[N,N],scale=scale)
        self.img=np.maximum(self.img+noise , 0)
        self.img=np.minimum(self.img       , 1)
    '''
    Regenerate picture For shape noise
    '''
    def S_noise(self, scale=0.02, num=0, back_ground="white"):
        N=len(self.img)
        temp=np.zeros([N,N])
        mid=int((N-1)/2)
        a_pixel=self.a*mid
        b_pixel=self.b*mid
        if(back_ground=="white"):
            fond=-1
        else:
            fond=1
            
        for i in range(N):
            for j in range(N):
                new_axis=self.rotation.dot(
                    np.array([[i-(self.center[0]+1)*mid],[j-(self.center[1]+1)*mid]]))
                
                x,y=new_axis[0][0],new_axis[1][0]
                
                rho=np.sqrt(x**2+y**2)+1e-15
                
                cos,sin=x*1.0/rho,y*1.0/rho
                #print(x,y,rho, r_pixel*(1-eps**2)/(1-eps*cos))
                theta=np.arccos(cos)
                
               
                                                      
                if((x-scale*a_pixel*np.cos(num*theta))**2/a_pixel**2 +(y-scale*a_pixel*np.sin(num*theta))**2/b_pixel**2<=1):
                    color=-1
                    temp[i,j]=(1-fond*color)/2
                else:
                    color=1
                    temp[i,j]=(1-fond*color)/2
        self.img=Fit_axis(temp)
    def Plot(self):
        plt.gray()
        my_dpi=5
        plt.figure(figsize=(27/my_dpi, 27/my_dpi), frameon=True)
        plt.imshow(self.img, interpolation='none')
    def Save_plot(self,pixel,save_name=""):
        my_dpi=20
        #change the pixel
        plt.figure(figsize=(pixel/my_dpi, pixel/my_dpi), frameon=True)
        plt.imshow(self.img, interpolation='none')
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        assert(save_name!=""),"File name empty!"
        plt.savefig(save_name+".png", bbox_inches='tight',pad_inches = 0,dpi=my_dpi)
        plt.close()
#%%
##Test            
#e=Eclipse(0.8,0.4,center=[0.0,0.2],rotation_degree=25,N=501)
#e.S_noise(num=50,scale=0.02)
#e.Plot()
#e.G_Noise(scale=0.3)
#e.Plot()
#e.Save_plot(200,"test")