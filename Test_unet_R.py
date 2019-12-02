
 
import glob
import os

import cv2
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
from tkinter.filedialog import askopenfilenames
from skimage import color
import Unet1   

K.set_image_data_format('channels_first')
K.set_learning_phase(0)

#from Unet1 import unet
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.layers import Input,Conv2D,MaxPooling2D,AveragePooling2D,BatchNormalization,Dropout,Conv2DTranspose,concatenate,Reshape,Permute,Activation, UpSampling2D
from keras.models import Model
# import NetPixel
import ResNetworks
dsc=[]
dsc_loss= []
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


data_path='/training/train_val/Blocks_1024_non_overlapping/train/small/'

path2write = '/training/train_val/Blocks_1024_non_overlapping/test/Train_out_Resnet/'
os.makedirs(path2write)

decoderPath = '/training/train_val/Blocks_1024_non_overlapping/model/500-363-0.98.h5'



imheight = 1024 
imwidth = 1024
imdepth = 3
data_shape = imheight * imwidth
classes = 2
data_format='.jpg'

dp=0

def testBatch():
    
    paths = sorted(glob.glob(data_path + "*.jpg"))  
  
    model = Unet1.unet(imwidth, imheight, imdepth, classes, decoderPath) 
  
    
    model.summary()

    for i in range(len(paths)):
        tdp = paths[i]            
        im = cv2.imread(tdp) 
        #im =  color.rgb2lab(im)
        #Compute the mean for data normalization
        b_ch=np.mean(im[:,:,0])
        g_ch=np.mean(im[:,:,1])
        r_ch=np.mean(im[:,:,2])  
#         # Mean substraction     
        im_ = np.array(im, dtype=np.float32)                             
        im_ -= np.array((b_ch,g_ch,r_ch))
        
        #compute the standard deviation
# =============================================================================
        b_ch=np.std(im[:,:,0])
        g_ch=np.std(im[:,:,1])
        r_ch=np.std(im[:,:,2])
        im_ /= np.array((b_ch,g_ch,r_ch))
# =============================================================================
        
        
        data = []
#         data.append(im)
        data.append(np.rollaxis((im_),2)) # it is for channel first
    
        temp = np.array(data)     
       
        prob = model.predict(temp, verbose= 1)        
      
        prediction = np.argmax(prob[0],axis=1)
      
     
        
        prediction = np.reshape(prediction,(imwidth,imheight))
        scale = np.uint8(255/(classes-1))
        norm_image = scale*np.uint8(prediction) 
       # norm_image = cv2.cvtColor(norm_image,CV_GRAY2RGB)       
        #norm_image = np.uint8(norm_image)                
        tilename = tdp.split('\\')
        tileNum = tilename[-1].split(data_format)
        tileNum= tileNum[0].split('/')
         
#         plt.imshow('output',norm_image)
#         cv2.waitKey(0)
        im_color = cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)

        cv2.imwrite(path2write+tileNum[-1]+'_modlabel.png',im_color)
#         cv2.imwrite(path2write+tileNum[-1]+'_modlabel.png',norm_image)
        
        print(tdp)
testBatch()
