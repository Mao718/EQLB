import tensorflow as tf
import numpy as np
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
def average_conv_layer(imgs:'(batch,width,width,colors)',width:int =4,stride=2,opt='flatten'):
    if opt=='flatten':
        number_of_pixel=width*width*imgs.shape[3]
        conv_kernal=np.ones((width,width,imgs.shape[3],1))
        return tf.nn.conv2d(imgs,conv_kernal,[1,stride,stride,1],padding='VALID') /number_of_pixel
def fig2seq(img:'(width,width)',frame_size=2
            ,stride=(1,1),decay_rate=0.9,switch=10)->"(time step,variables)":

    wide=int((img.shape[0]-(frame_size-stride[0]))/stride[0]) #k
    hight=int((img.shape[1]-(frame_size-stride[1]))/stride[1]) #k
    img_seq=np.zeros((switch*wide*hight,img.shape[0],img.shape[1]))
    #print('sequence_len=',switch*wide*hight)
    for t in range(img_seq.shape[0]):
        if t%switch==0:

            h=int(t/(wide*switch))
            residue=int(t%(wide*switch))
            w=int(residue/switch)
            img_seq[t,h:h+frame_size,w:w+frame_size]=np.clip(img_seq[t,h:h+frame_size,w:w+frame_size]
                                                             +img[h:h+frame_size,w:w+frame_size],0,1)

        else:
            img_seq[t]=np.clip(img_seq[t-1]*decay_rate,0,1)


    return img_seq.reshape(img_seq.shape[0],img_seq.shape[1]*img_seq.shape[2])
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print(average_conv_layer(x_train/255).shape)