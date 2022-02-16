from preprocess.preprocess import *
from embedding.embeddings import *
from evaluation.evaluate import evaluate
import tensorflow as tf
import numpy as np

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
data='cifar10'
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
#x_use = x_train[:5000]
#y_use = y_train[:5000]



#conv1
#------training set
conv=average_conv_layer(x_train)
conv=tf.squeeze(conv)
rep_train=mutiprocess_embedding(conv)
rep_train=np.array(rep_train)
#-------testing set 
conv_test=average_conv_layer(x_test)
conv_test=tf.squeeze(conv_test)
rep_test=mutiprocess_embedding(conv_test)
rep_test=np.array(rep_test)

for i in range(10):
    evaluate(rep_train.reshape(
        60000, -1), rep_test.reshape(10000, -1), y_train, y_test, i+1)
    
        