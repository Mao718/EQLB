import sys
sys.path.append("..")

from multiprocessing import Pool
import multiprocessing
import tqdm

from model.syncmap import SyncMap
from preprocess.preprocess import *

def work_package_for_mutiprocess(img:"(width,width,color)",repeat_time=1):
    seq=fig2seq(img)
    model=SyncMap(seq.shape[1], 3)
    for i in range(repeat_time):
        rep=model.input(seq)
    return rep

def mutiprocess_embedding(img:"(batch,width,width,color)"):
    with Pool(multiprocessing.cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(work_package_for_mutiprocess, img)
                           , total=img.shape[0]))
    return r
    
if __name__ == '__main__':
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    a=average_conv_layer(x_train[:100]/255)
    a=tf.squeeze(a)
    rep=mutiprocess_embedding(a)
    
    
    seq=fig2seq(a[0])
    model=SyncMap(seq.shape[1], 3)
    test_rep=model.input(seq)
    
    