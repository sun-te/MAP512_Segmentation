# MAP512_Segmentation
Theses are codes for a short project during a class MAP512 at Ecole polytechnique. 2019, Second term

## Random Walker
```Random walker``` algorithm is included in the directory **random_walker**, one can test the algorithm with the script **test.py**

## AtlasNet by transfer learning
Please use the ```notebook``` 	```Transfer Learning.ipynb``` in the folder ```Segnet``` to train our tranfer learning model 
and see the results

## Data generator
At the same time, one can find the data-generator in the file **ImgGenerator.py**. It will create images like

<div align="center">
    <img src="random_walker/data/noise/Noise_0.png", width="200">
</div>
We can absolutely change the standard deviation (scale of the gaussian noise) and the number of shape deformation in the code


## A brief comparison of the performance of RW and SegNet

From left to right: the original image to be segmented, ground truth label, result by  **Random walker**  and  **SegNet** 

<div align="center">
    <img src="SegNet/Result/original_noisy.png", width="200">
    <img src="SegNet/Result/0005.png", width="200">
    <img src="SegNet/Result/random_walker_beta%3D90.png", width="200">
    <img src="SegNet/Result/Segnet.png", width="200">
</div>
