#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:21:23 2019

@author: tete
"""
from Segnet_transfer import SegNet   
import keras
from Scripts import Random_walker,EclipseGenerator
from keras.models import Sequential
from keras.layers import Conv2D,ReLU,BatchNormalization,LeakyReLU,Reshape
from keras import models, layers
import tensorflow as tf
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler
from imageio import imread
from inputs_object import get_filename_list
#%%

num_classes=2
def one_hot(y):
    
    length,w,h,c=y.shape
    output=[]

    for i in range(length):
        x=y[i].reshape([1,-1]).astype(int)
        tmp=np.zeros([ num_classes,w*h])
        tmp[0]=1-x
        tmp[1]=x
        tmp=tmp.T
        output.append(tmp)
    
    return np.array(output)
#%%
    

'''Loading data'''
seg=SegNet()    
logits,labels, images=seg.transfer_output(dataset_type = "TRAIN")
logits=np.array([m[0][0] for m in logits])
labels=np.array([m[0] for m in labels])
images=np.array([m[0] for m in images])
train_images=images
x_train,y_train=logits,one_hot(labels/255)

seg=SegNet()
logits,labels, images=seg.transfer_output(dataset_type = "TEST")
logits=np.array([m[0][0] for m in logits])
labels=np.array([m[0] for m in labels])
images=np.array([m[0] for m in images])
test_images=images
x_test,y_test=logits,one_hot(labels/255)
#%%
'''transfer learning'''
def model(x):
    x=layers.Conv2D(filters=32,input_shape=(128,128,12), kernel_size=(3,3),padding='same')(x)
    x=layers.LeakyReLU()(x)
    x=layers.BatchNormalization()(x)
    
    x=layers.Conv2D(filters=16, kernel_size=(3,3),padding='same')(x)
    x=layers.LeakyReLU()(x)
    x=layers.BatchNormalization()(x)
    
    
    x=layers.Conv2D(filters=num_classes, 
                     kernel_size=(3,3),padding='same',activation='softmax')(x)
    
    x=Reshape([-1,2])(x)
    return x
image_tensor = layers.Input(shape=(128,128,12))
network_output = model(image_tensor)    
model = models.Model(inputs=[image_tensor], outputs=[network_output])

model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
print(model.summary())

#%%
batch_size = 10
epochs = 50
def scheduler(epoch):
    if epoch % 50==0 and epoch<800 and epoch >1:
      K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr)*0.1)
    return K.get_value(model.optimizer.lr)

change_lr = LearningRateScheduler(scheduler)
# Run the train
history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[change_lr])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#%%
plt.figure(figsize=(7, 5))
plt.plot(history.epoch, history.history['acc'], lw=3, label='Training')
plt.plot(history.epoch, history.history['val_acc'], lw=3, label='Testing')
plt.legend(fontsize=14)
plt.title('Accuracy of CNN', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.tight_layout()
#%%
'''comparaison of the results'''
no=8
m=model.predict(x_test)
res=value_max = np.argmax(m[no],1)
res=res.reshape(128,128)
plt.imshow(test_images[no])
plt.show()
plt.imshow(res)
print(res)


score = model.evaluate(np.array([x_test[no]]), np.array([y_test[no]]), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#%%

test_file=get_filename_list(seg.test_file,seg.config)
la_filename=test_file[0][no]

im = imread(la_filename,pilmode='L')/255
e=EclipseGenerator.Eclipse(1,1,N=128)

e.img=im
e.Plot()
plt.show()


seeds=[[int(len(im[no])/2),int(len(im[no])/2)],[127,127]]
labels=[0,1]
beta=90
[mask,proba]=Random_walker.random_walker(im,seeds,labels,beta)
e.img=mask
e.Plot()
#%%

def loss(x):
    #x=Reshape([-1,2])(x)
    return x
image_tensor = layers.Input(shape=(128*128,2))
network_output = loss(image_tensor)    
loss = models.Model(inputs=[image_tensor], outputs=[network_output])

loss.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
print(loss.summary())
mask_onehot=np.array([1-mask.reshape([128*128,1]),mask.reshape([128*128,1])]).T

#%%
score = loss.evaluate(mask_onehot, np.array([y_test[no]]), verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%
#
#
#
#
#def get_filename_list(path, config):
#    fd = open(path)
#    image_filenames = []
#    label_filenames = []
#    for i in fd:
#        i = i.strip().split(" ")
#        image_filenames.append(i[0])
#        label_filenames.append(i[1])
#
#    image_filenames = [config["IMG_PREFIX"] + name for name in image_filenames]
#    label_filenames = [config["LABEL_PREFIX"] + name for name in label_filenames]
#    return image_filenames, label_filenames
#
#
#def get_all_test_data(im_list, la_list):
#    images = []
#    labels = []
#    index = 0
#    for im_filename, la_filename in zip(im_list, la_list):
#        im = imread(im_filename,pilmode='RGB')
#        la = imread(la_filename,pilmode='L')
#        images.append(im)
#        labels.append(la)
#        index = index + 1
#
#    print('%d CamVid test images are loaded' % index)
#    return images, labels
#
#
#
#def cal_loss(logits, labels,number_class):
#    loss_weight = np.array([
#        0.2595,
#        0.1826,
#        4.5640,
#        0.1417,
#        0.9051,
#        0.3826,
#        9.6446,
#        1.8418,
#        0.6823,
#        6.2478,
#        7.3614,
#        1.0974
#    ])
#    loss_weight = np.ones(shape=12)
#    
#    
#    labels = tf.to_int64(labels)
#    loss, accuracy, prediction = weighted_loss(logits, labels, number_class=number_class, frequency=loss_weight)
#    
#    return loss, accuracy, prediction
#
#
#def weighted_loss(logits, labels, number_class, frequency):
#    label_flatten = tf.reshape(labels, [-1])
#    
#    #batch size * number_class 
#    label_onehot = tf.one_hot(label_flatten, depth=number_class)
#    
#    #batch_size*64*64*number_class -> batch size * number_class 
#    logits_reshape = tf.reshape(logits, [-1, number_class])
#    
#    '''
#    A value pos_weights > 1 decreases the false negative count, 
#    hence increasing the recall. Conversely setting pos_weights < 1 
#    decreases the false positive count and increases the precision. 
#    This can be seen from the fact that pos_weight is introduced 
#    as a multiplicative coefficient for the positive targets term 
#    in the loss expression:
#    如果小于1，那么我就尽可能不把正确的分错，增加权重
#    比如行人，我是非常不想分错的
#    
#    targets * -log(sigmoid(logits)) * pos_weight +
#    (1 - targets) * -log(1 - sigmoid(logits))'''
#    
#    #batch num *num class
#    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
#            targets=label_onehot, logits=logits_reshape,pos_weight=frequency)
#    
#    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#    
#    #add cross entropy mean in to the summary and as a scaler
#    tf.summary.scalar('loss', cross_entropy_mean)
#    
#    #return element wise 
#    correct_prediction = tf.equal(tf.argmax(logits_reshape, -1), label_flatten)
#    #print("*********************************")
#    #print(label_flatten.shape)
#    #print("*********************************")
#    accuracy = tf.reduce_mean(tf.dtypes.cast(correct_prediction,tf.float32))
#    tf.summary.scalar('accuracy', accuracy)
#    
#    #return with the biggest probability
#    return cross_entropy_mean, accuracy, tf.argmax(logits_reshape, -1)
#
#
#def loss(logits, labels, number_class,frequency):
#    label_flatten = tf.reshape(labels, [-1])
#    
#    #batch size * number_class 
#    label_onehot = tf.one_hot(label_flatten, depth=number_class)
#    
#    #batch_size*64*64*number_class -> batch size * number_class 
#    logits_reshape = tf.reshape(logits, [-1, number_class])
#    
#    '''
#    A value pos_weights > 1 decreases the false negative count, 
#    hence increasing the recall. Conversely setting pos_weights < 1 
#    decreases the false positive count and increases the precision. 
#    This can be seen from the fact that pos_weight is introduced 
#    as a multiplicative coefficient for the positive targets term 
#    in the loss expression:
#    如果小于1，那么我就尽可能不把正确的分错，增加权重
#    比如行人，我是非常不想分错的
#    
#    targets * -log(sigmoid(logits)) * pos_weight +
#    (1 - targets) * -log(1 - sigmoid(logits))'''
#    
#    #batch num *num class
#    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
#            targets=label_onehot, logits=logits_reshape,pos_weight=frequency)
#    
#    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#    
#    #add cross entropy mean in to the summary and as a scaler
#    tf.summary.scalar('loss', cross_entropy_mean)
#    
#    #return element wise 
#    correct_prediction = tf.equal(tf.argmax(logits_reshape, -1), label_flatten)
#    #print("*********************************")
#    #print(label_flatten.shape)
#    #print("*********************************")
#    accuracy = tf.reduce_mean(tf.dtypes.cast(correct_prediction,tf.float32))
#    tf.summary.scalar('accuracy', accuracy)
#    
#    #return with the biggest probability
#    return cross_entropy_mean, accuracy, tf.argmax(logits_reshape, -1)

#%%

#%%
#%%

images_index=[0,1]
conf_file="config2.json"
seg=SegNet(conf_file="config2.json")
train_dir = seg.config["SAVE_MODEL_DIR"]
image_w = seg.config["INPUT_WIDTH"]
image_h = seg.config["INPUT_HEIGHT"]
image_c = seg.config["INPUT_CHANNELS"]
train_dir = seg.config["SAVE_MODEL_DIR"]
FLAG_BAYES = seg.config["BAYES"]
FLAG_MAX_VOTE = False


with seg.sess as sess:
    saver=tf.train.Saver()
    saver.restore(sess, train_dir)
    #self.logits=tf.nn.bias_add(self.conv, self.biases, name=scope.name)
    _, _, prediction = cal_loss(logits=seg.logits,labels=seg.labels_pl,number_class=seg.num_classes)
    
    prob = tf.nn.softmax(seg.logits,dim = -1)
    test_type_path = seg.config["TRAIN_FILE"]
    indexes = images_index
    image_filename,label_filename = get_filename_list(test_type_path, seg.config)
    images, labels = get_all_test_data(image_filename,label_filename)
    
    #images = [images[i] for i in indexes]
    #labels = [labels[i] for i in indexes]
    

    image_batch=np.reshape(images,[len(images),image_h,image_w,image_c])
    label_batch=np.reshape(labels,[len(labels),image_h,image_w,1])
    pred_tot = []
    var_tot = []
    logit_tot=[]

    for image_batch, label_batch in zip(images,labels):
        print(image_batch.shape)
        #to just convert the image into the standard format
        #for example: 1(batch size),64,64,3
        image_batch = np.reshape(image_batch,[1,image_h,image_w,image_c])
        label_batch = np.reshape(label_batch,[1,image_h,image_w,1])
        #non baysien model
        fetches = [prediction]
        feed_dict = {seg.inputs_pl: image_batch,
                     seg.labels_pl: label_batch,
                     seg.is_training_pl: False,
                     seg.keep_prob_pl: 0.5,
                     seg.with_dropout_pl: True,
                     seg.batch_size_pl: 1}
        pred = sess.run(fetches = fetches, feed_dict = feed_dict)
        logit= sess.run([seg.logits],feed_dict = feed_dict)
        #原本是images_h* images_w, with each element the label of it
        pred = np.reshape(pred,[image_h,image_w])
        var_one = []
        pred_tot.append(pred)
        var_tot.append(var_one)
        logit_tot.append(logit)
