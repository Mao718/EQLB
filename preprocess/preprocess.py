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

def slice_img(img:'(width,width,color)',frame_size=8
            ,stride=(4,4))->"(time step,variables)":
    wide=int((img.shape[0]-(frame_size-stride[0]))/stride[0]) #k
    hight=int((img.shape[1]-(frame_size-stride[1]))/stride[1]) #k
    img_list=[]
    for i in range(wide):
        for j in range(hight):
            img_list.append(img[stride[0]*i:stride[0]*i+frame_size,stride[0]*j:stride[0]*j+frame_size])
    return np.array(img_list)
def slice_img3D(img:'(width,width,color)',frame_size=8
            ,stride=(4,4))->"(time step,variables)":
    wide=int((img.shape[0]-(frame_size-stride[0]))/stride[0]) #k
    hight=int((img.shape[1]-(frame_size-stride[1]))/stride[1]) #k
    slice_img=np.zeros((wide,hight,img.shape[2]))
    
    for i in range(wide):
        for j in range(hight):
            #print(np.mean(img[stride[0]*i:stride[0]*i+frame_size,stride[0]*j:stride[0]*j+frame_size],axis=2).shape)
            slice_img[i,j]=img[stride[0]*i:stride[0]*i+frame_size,stride[0]*j:stride[0]*j+frame_size].mean(axis=0).mean(axis=0)
    return slice_img
def fig2seq_3D(img:'(width,width,hight)',frame_size=2
            ,stride=(1,1),decay_rate=0.9,switch=10)->"(time step,variables)":

    wide=int((img.shape[0]-(frame_size-stride[0]))/stride[0]) #k
    hight=int((img.shape[1]-(frame_size-stride[1]))/stride[1]) #k
    img_seq=[]
    img_now=np.zeros((img.shape))
    #print(img_now.shape,'<---------')
    #print('sequence_len=',switch*wide*hight)
    for i in range(wide):
        for j in range(hight):
            decay_time=0
            #print(stride[0]*i,'-->',stride[0]*i+frame_size)
            while decay_time<=switch:
                
                if decay_time==0: 
                    img_now[stride[0]*i:stride[0]*i+frame_size,stride[0]*j:stride[0]*j+frame_size]+=img[stride[0]*i:stride[0]*i+frame_size,stride[0]*j:stride[0]*j+frame_size]
                img_seq.append(img_now.copy())
                img_now*=0.9
                
                decay_time+=1
    img_seq=np.array(img_seq)

    return img_seq.reshape(img_seq.shape[0],img_seq.shape[1]*img_seq.shape[2]*3) #,img_seq   
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    #test,test_use=fig2seq_3D(x_train[0]/255.)
    test=slice_img3D(x_train[0]/255.)
    import matplotlib.animation as animation
    fig=plt.figure()
    ims=[]
    for a in range(test_use.shape[0]):
        ims.append([plt.imshow(test_use[a])])
    ani = animation.ArtistAnimation(fig, ims, interval=50,repeat=False)
    print("saving")
    mywriter = animation.FFMpegWriter()
    ani.save('dynamic_images.mp4',writer=mywriter)