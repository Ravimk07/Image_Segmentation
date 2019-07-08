
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras import callbacks  
# from keras.utils.visualize_util import plot
import keras.backend as K
K.set_image_data_format("channels_first")
from keras.layers import Convolution2D,Permute,Activation,Reshape,ZeroPadding2D
from keras.optimizers import RMSprop,SGD
import Unet 

#from Networks import Resnet_Modified
# import Resnet_Modified
import numpy as np
from keras.layers.convolutional import UpSampling2D

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


np.random.seed(1337) # for reproducibility

train_data_path = '/mnt/x/Ravi K/New_POC_data/Images/'
train_label_path = '/mnt/x/Ravi K/New_POC_data/Label2/'

# pretrainedPath = 'E:/AvinashLokhande/PyDevWorkSpace/DeepLearningTF/Ovaries/Models/AutoEncoderSkin_19.h5'
# path2save = 'E:/AvinashLokhande/PyDevWorkSpace/DeepLearningTF/MammaryGland/Models/ResNet50_Deconv_G.h5'
 
path2save = '/home/ravik/eclipse-workspace/POC/POC_para1.h5'


imheight = 512
imwidth = 512
imdepth = 3
data_shape = imheight*imwidth
classes = 2

train_data = np.load(train_data_path+'data.npy')
# train_data  = train_data .astype("float32")
# train_data  = train_data/255.0

train_label = np.load(train_label_path+'label.npy')
train_label = np.reshape(train_label,(len(train_label),data_shape,classes))

net = Unet.get_unet() 
net.summary()


modelCheck = callbacks.ModelCheckpoint(path2save, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')

TB_LOG = '/home/ravik/eclipse-workspace/POC/log_para_2c'  
 
tensorboard = callbacks.TensorBoard(log_dir=TB_LOG,
                                  histogram_freq=0,
                                  batch_size=8,
                                  write_graph=True,
                                  write_grads=False,
                                  write_images=False,
                                  embeddings_freq=0,
                                  embeddings_layer_names=None,
                                  embeddings_metadata=None)
# optimizer = SGD(lr=1e-15, momentum=0.9, decay=0.0, nesterov=False)
optimizer = RMSprop(lr=1e-5, rho=0.95, epsilon=1e-8, decay=0)

print ("Training the Model...")

# net.fit(train_data, train_label, batch_size = 1, epochs = 50, verbose=2,callbacks= [modelCheck],validation_split=0.2)
net.fit(train_data, train_label, batch_size= 8, epochs = 20, verbose=2,callbacks= [modelCheck,tensorboard])

path2save1 = '/home/ravik/eclipse-workspace/POC/POC_para_2c.h5'
print ("Dumping Weights to file...")
net.save_weights(path2save1, overwrite=True)
print ("Models Saved :)")
