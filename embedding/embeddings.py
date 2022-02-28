import sys
sys.path.append("..")

from multiprocessing import Pool
import multiprocessing
import tqdm
from model.sigma import Sigma
from model.syncmap import SyncMap
from preprocess.preprocess import *

def work_package_for_mutiprocess(img:"(width,width,color)",repeat_time=10):
    
    seq=fig2seq(img)
    #print(seq.shape)
    model=SyncMap(seq.shape[1], 2)
    for i in range(repeat_time):
        rep=model.input(seq)
    #rep=model.syncmap
    return rep

def mutiprocess_embedding(img:"(batch,width,width,color)"):
    with Pool(multiprocessing.cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(work_package_for_mutiprocess, img)
                           , total=img.shape[0]))
    return r

def work_package_for_mutiprocess_3D(img:"(width,width,color)",repeat_time=10):
    img=slice_img3D(img)
    seq=fig2seq_3D(img)
    #print(seq.shape)
    model=SyncMap(seq.shape[1], 2)
    for i in range(repeat_time):
        rep=model.input(seq)
    #rep=model.syncmap
    return rep

def mutiprocess_embedding_3D(img:"(batch,width,width,color)"):
    with Pool(multiprocessing.cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(work_package_for_mutiprocess_3D, img)
                           , total=img.shape[0]))
    return r






#------------------testing only
def work_package_for_mutiprocess_test(seq:"(time variable)",repeat_time=10):
    model=SyncMap(seq.shape[1], 2)
    for i in range(repeat_time):
        rep=model.input(seq)
    #rep=model.syncmap
    return rep
def mutiprocess_embedding_test(img:"(batch,width,width,color)"):
    with Pool(multiprocessing.cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(work_package_for_mutiprocess_test, img)
                           , total=len(img)))
    return r
"""--------------------------------------------
import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
#from self_organized_neurons.Sigma import*
#from self_organized_neurons.SyncMap import*
import time
from tqdm import trange
import tqdm
from tqdm.contrib import tenumerate
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib as mpl
from multiprocessing import Pool
# mutiprocess tool
from functools import partial
import os
from torch.utils.data import DataLoader
#from syncmap_gpu import batch_inputGeneral
#from syncmap_cpu import *


class EQLB():
    def __init__(self, representation_dim):
        self.representation_dim = representation_dim
        self.is_w_init = False

    def preprocess_data_conv(self, data, frame_size=4, stride=2, pool='mean'):
        frame_number = int((data.shape[1]-frame_size/2)/stride)
        print("frame_number = ", frame_number)
        frame_data = np.zeros(
            (data.shape[0], frame_number, frame_number, frame_size, frame_size, 1))
        for i in range(frame_number):
            for j in range(frame_number):
                frame_data[:, i, j] = data[:, int(
                    i*stride):int(i*stride+frame_size), int(j*stride):int(j*stride+frame_size)]
        print("frame_data.shape", frame_data.shape)
        frame_data = np.squeeze(frame_data)
        print("frame_data.shape-------", frame_data.shape)
        if pool == 'sum':
            frame_data = np.sum(frame_data, axis=3)
            frame_data = np.sum(frame_data, axis=3)
            self.X = frame_data
        if pool == 'mean':
            frame_data = np.mean(frame_data, axis=3)
            frame_data = np.mean(frame_data, axis=3)
            self.X = frame_data

    def fig2seq(self, img, frame_size=2, stride=(1, 1), decay_rate=0.9, switch=10):

        wide = int((img.shape[0]-(frame_size-stride[0]))/stride[0])  # k
        hight = int((img.shape[1]-(frame_size-stride[1]))/stride[1])  # k
        img_seq = np.zeros((switch*wide*hight, img.shape[0], img.shape[1]))
        # print('sequence_len=',switch*wide*hight)
        for t in range(img_seq.shape[0]):
            if t % switch == 0:

                h = int(t/(wide*switch))
                residue = int(t % (wide*switch))
                w = int(residue/switch)
                img_seq[t, h:h+frame_size, w:w+frame_size] = np.clip(img_seq[t, h:h+frame_size, w:w+frame_size]
                                                                     + img[h:h+frame_size, w:w+frame_size], 0, 1)
            else:
                img_seq[t] = np.clip(img_seq[t-1]*decay_rate, 0, 1)
        return img_seq.reshape(img_seq.shape[0], img_seq.shape[1]*img_seq.shape[2])

    def data_preprocess(self, data):
        print('data is preprocessing')
        self.preprocess_data_conv(data)
        self.datasets = []
        if not self.is_w_init:
            self.w_init = np.random.rand(
                self.X.shape[1]*self.X.shape[2], self.representation_dim)
            self.is_w_init = True
        for i in trange(self.X.shape[0]):
            self.datasets.append(self.fig2seq(self.X[i]))
        return self.datasets

    def encoder(self, data, batch_size, repeat=10):
        self.data_preprocess(data)
        legth = int(len(self.datasets)/batch_size)

        data_loader = DataLoader(
            self.datasets, batch_size=batch_size, shuffle=False)
        for i, data in tenumerate(data_loader):
            # init all the batch at the same time
            w_init = torch.tensor(self.w_init)
            w_init = torch.unsqueeze(w_init, 0).repeat(batch_size, 1, 1)
            representation = w_init
            # print('representation',representation)
            # init the image with batch
            if i == 0:
                self.outputs = self.batch_SyncMap(
                    data.repeat(1, repeat, 1), representation)
            else:
                self.outputs = np.concatenate((self.outputs,
                                               self.batch_SyncMap(data.repeat(1, repeat, 1), representation)), 0)

    def batch_SyncMap(self, data, w_init, lr=0.01, space_size=10):
        # data(batch,seq,states)
        data = data.cpu().numpy()
        self.representation = w_init.cpu().numpy()
        plus = data > 0.1
        minus = data <= 0.1
        # the batch witch didn't start training will not be normalize
        self.start_normalize = np.zeros(data.shape[0], dtype=bool)
        for i in range(data.shape[1]):
            # print(i)
            vplus = plus[:, i, :]
            vminus = minus[:, i, :]  # (batch, states)
            # print('vminus',vminus)
            # this is for quciker train skip if activate point is not enough
            continue_condition_P = vplus.sum()
            continue_condition_N = vminus.sum()
            #print(i,'th continue condition',continue_condition_P,continue_condition_N)
            if continue_condition_P <= 1:
                # print('continue')
                continue
            if continue_condition_N <= 1:
                # print('continue')
                continue

            # some of the batch contain only one activate point need to be remove
            plus_mass = vplus.sum(axis=1)  # (batch)
            plus_mass[plus_mass == 1] = 0
            minus_mass = vminus.sum(axis=1)  # (batch)
            # also if there're no acitivte points should not have nagetive points
            minus_mass = np.where(plus_mass == 0, 0, minus_mass)

            # --------------------------------used for true center_plus mask
            mask_plus = plus_mass.copy()
            mask_plus[mask_plus > 1] = 1
            mask_plus = np.expand_dims(mask_plus, 1)
            mask_plus = np.expand_dims(mask_plus, 2)
            mask_plus = np.tile(
                mask_plus, [1, self.representation.shape[1], self.representation.shape[2]])
            #print(i,'th mask plus',mask_plus)
            mask_minus = minus_mass.copy()
            mask_minus[mask_minus > 1] = 1
            mask_minus = np.expand_dims(mask_minus, 1)
            mask_minus = np.expand_dims(mask_minus, 2)
            mask_minus = np.tile(
                mask_minus, [1, self.representation.shape[1], self.representation.shape[2]])

            # which set the mass who's 0 to 1 to prevent devide 0
            non_zero_plus_mass = plus_mass.copy()
            non_zero_plus_mass[non_zero_plus_mass == 0] = 1  # (batch)
            non_zero_minus_mass = minus_mass.copy()
            non_zero_minus_mass[non_zero_minus_mass == 0] = 1
            #representation (batch,state,dim)

            # parameter used for center calculation-------
            center_plus_temp_map = np.tile(np.expand_dims(vplus, axis=2), [
                                           1, 1, self.representation_dim])
            center_minus_temp_map = np.tile(np.expand_dims(vminus, axis=2), [
                                            1, 1, self.representation_dim])
            center_plus_mass_temp = np.expand_dims(non_zero_plus_mass, 1)
            center_plus_mass_temp = np.tile(
                center_plus_mass_temp, [1, self.representation.shape[2]])
            # (batch,dim)
            center_minus_mass_temp = np.expand_dims(non_zero_minus_mass, 1)
            center_minus_mass_temp = np.tile(center_minus_mass_temp, [
                                             1, self.representation.shape[2]])
            # center_minus_mass_temp=np.tile(center_minus_mass_temp,self.representation.shape[2],axis=1)
            # ----------------------------
            center_plus = ((mask_plus*center_plus_temp_map *
                           self.representation).sum(axis=1)/center_plus_mass_temp)
            center_minus = ((mask_minus*center_minus_temp_map *
                            self.representation).sum(axis=1)/center_minus_mass_temp)

            cdist_plus_t = np.tile(np.expand_dims(center_plus, 1), [
                                   1, self.representation.shape[1], 1])
            cdist_minus_t = np.tile(np.expand_dims(center_minus, 1), [
                                    1, self.representation.shape[1], 1])
            # cdist_plus_t=np.tile(np.expand_dims(center_plus,1),self.representation.shape[1],axis=1)
            # cdist_minus_t=np.tile(np.expand_dims(center_minus,1),self.representation.shape[1],axis=1)

            # the distant between positve and negative center to all other points
            cdist_plus = np.sqrt(
                np.square(self.representation-cdist_plus_t).sum(axis=2))  # batch, state
            cdist_minus = np.sqrt(
                np.square(self.representation-cdist_minus_t).sum(axis=2))
            # print('cdist_plus',cdist_plus)
            # print('cdist_minus',cdist_minus)
            # prevent devide 0---------
            cdist_plus[cdist_plus == 0] = 1
            cdist_minus[cdist_minus == 0] = 1
            # -------------------
            # batch, state dim
            # the parameter used for update-----------------
            center_plus_temp = np.expand_dims(center_plus, 1)
            center_plus_temp = np.tile(center_plus_temp,
                                       [1, self.representation.shape[1], 1])
            # center_plus_temp=np.tile(np.expand_dims(center_plus,1),self.representation.shape[1],axis=1)
            # center_minus_temp=np.tile(np.expand_dims(center_minus,1),self.representation.shape[1],axis=1)
            center_minus_temp = np.expand_dims(center_minus, 1)
            center_minus_temp = np.tile(center_minus_temp,
                                        [1, self.representation.shape[1], 1])

            cd_update_plus_temp = np.tile(np.expand_dims(cdist_plus, 2), [
                                          1, 1, self.representation.shape[2]])
            cd_uqdate_minus_temp = np.tile(np.expand_dims(cdist_minus, 2), [
                                           1, 1, self.representation.shape[2]])

            # cd_update_plus_temp=np.tile(np.expand_dims(cdist_plus,2),self.representation.shape[2],axis=2)
            # cd_uqdate_minus_temp=np.tile(np.expand_dims(cdist_minus,2),self.representation.shape[2],axis=2)
            #
            plus_temp = np.tile(np.expand_dims(vplus, 2), [
                                1, 1, self.representation.shape[2]])
            minus_temp = np.tile(np.expand_dims(vminus, 2), [
                                 1, 1, self.representation.shape[2]])
            # plus_temp=np.tile(np.expand_dims(vplus,2),self.representation.shape[2],axis=2)
            # minus_temp=np.tile(np.expand_dims(vminus,2),self.representation.shape[2],axis=2)
            # --------------------------------used for true center_plus mask-
            update_plus = mask_plus*center_plus_temp_map * \
                ((center_plus_temp-self.representation)/cd_update_plus_temp)
            update_minus = mask_minus*center_minus_temp_map * \
                ((center_minus_temp-self.representation)/cd_uqdate_minus_temp)
            update = update_plus-update_minus
            temp = update.sum(axis=2)
            temp = temp.sum(axis=1)  # batch
            self.start_normalize[temp != 0] += bool(1)
            # print(i,'--th--update',update)
            self.representation += lr*update
            # normalization-------------------
            maximun = self.representation.max(axis=2).max(axis=1)  # batch
            normalize = space_size/maximun
            normalize[self.start_normalize == 0] = 1
            normalize = np.expand_dims(normalize, 1)
            normalize = np.expand_dims(normalize, 2)
            normalize = np.tile(
                normalize, [1, self.representation.shape[1], self.representation.shape[2]])
            # maximun=np.tile(maximun,self.representation.shape[1],axis=1)
            # maximun=np.tile(maximun,self.representation.shape[2],axis=2)
            self.representation = self.representation*normalize

        return self.representation

    def evaluate(self, train_representation, test_representation, train_label, test_label, neighbors=5):
        classifier = KNeighborsClassifier(n_neighbors=neighbors)
        classifier.fit(train_representation, train_label)
        y_pred = classifier.predict(test_representation)
        print('accuracy =', accuracy_score(test_label, y_pred))"""

"""--------------------------------------------"""    
if __name__ == '__main__':
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    #------
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    #x_train=np.squeeze(x_train) ####if cifar 10

    x_use = x_train[:5000]
    y_use = y_train[:5000]
    
    #y_use=np.squeeze(y_use)
    #    useable
    #Model = EQLB(2)
    #data = Model.data_preprocess(x_use)
    #rep=mutiprocess_embedding_test(data)
    #---------------------
    
    "---------"
    a=average_conv_layer(x_use)
    a=tf.squeeze(a)
    rep=mutiprocess_embedding(a)
    #
    "----------"
    #print(seq[500, :])
    rep=np.array(rep)
    for i in range(10):
        evaluate(rep[:4000].reshape(
            4000, -1), rep[4000:].reshape(1000, -1), y_use[:4000], y_use[4000:], i+1)
        
        
    #seq=fig2seq(a[0])
    #model=SyncMap(seq.shape[1], 3)
    #test_rep=model.input(seq)
    
    
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    #x_use = x_train[:5000]
    #y_use = y_train[:5000]
    
    #pre1=average_conv_layer(x_use)
    #pre2=[]
    #for i in range(pre1.shape[0]):
    #    pre2.append(fig2seq(pre1[i]))
    
    
    