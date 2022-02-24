from preprocess.preprocess import *
from embedding.embeddings import *
from evaluation.evaluate import evaluate
from plot.plots import embedding_plot
import tensorflow as tf
import numpy as np

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
data='cifar10'
plt_logic=True
#load data mnist----------------------------
if data=='mnist':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

if data=='cifar10':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    x_train=np.squeeze(x_train)
    x_test=np.squeeze(x_test)
    y_train=np.squeeze(y_train)
    y_test=np.squeeze(y_test)





######version 1----------------------------

#conv=average_conv_layer(x_train)
#conv=tf.squeeze(conv)
#conv_test=average_conv_layer(x_test)
#conv_test=tf.squeeze(conv_test)
#rep_train=mutiprocess_embedding(conv_train)
#rep_test=mutiprocess_embedding(conv_test)
#rep_train=np.array(rep_train)
#rep_test=np.array(rep_test)
######--------------------------------------

######version 2 slice----------------------------

conv_train=[]
conv_test=[]
for i in range(x_train.shape[0]):
    conv_train.append(slice_img(x_train[i]))
for i in range(x_test.shape[0]):
    conv_test.append(slice_img(x_test[i]))
conv_train=np.array(conv_train)
conv_test=np.array(conv_test)

conv_train=conv_train.reshape(50000*49,8,8,3)
#conv_train=conv_train.mean(axis=3)

conv_test=conv_test.reshape(10000*49,8,8,3)

rep_train=mutiprocess_embedding_3D(conv_train)
rep_test=mutiprocess_embedding_3D(conv_test)
#conv_test=conv_test.mean(axis=3)

rep_train=np.array(rep_train)
rep_train=rep_train.reshape(50000,49,-1)

rep_test=np.array(rep_test)
rep_test=rep_test.reshape(10000,49,-1)
######---------------------------------------
for i in range(20):
    evaluate(rep_train.reshape(
        50000, -1), rep_test.reshape(10000, -1), y_train, y_test, i+1)
if plt_logic:
    embedding_plot(rep_train,y_train)
        