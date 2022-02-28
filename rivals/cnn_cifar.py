import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
import sys
sys.path.append("..")
from plot.plots import embedding_plot


def preprocess():
    (x_train, y_train), (x_test, y_test)=tf.keras.datasets.cifar10.load_data()
    print("y_train_shape:",y_train.shape)
    y_train=y_train.reshape(50000)
    x_train_normalize =x_train.astype('float32') / 255.0
    x_test_normalize = x_test.astype('float32') / 255.0
    y_train_onehot=tf.one_hot( y_train,10)
    y_test_onehot=tf.one_hot( y_test,10)
    
    print("y_train_onehot_shape:",y_train_onehot.shape)
    
    #return x_train_normalize,x_test_normalize,y_train_onehot, y_test_onehot
    return x_train_normalize,x_test_normalize,y_train, y_test,y_train_onehot, y_test_onehot

def show_train_history(history,train,validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()



if __name__ == '__main__':
    x_train_normal,x_test_normal,y_train,y_test,y_train_onehot,y_text_onehot=preprocess()
    
    #model
    inputs = keras.Input(shape=x_train_normal.shape[1:])
    
    x= layers.Conv2D(10,kernel_size=(4,4),strides=(2,2),padding='valid',activation='relu')(inputs)
    #x= layers.Conv2D(10,kernel_size=(4,4),strides=(2,2),padding='valid',activation='relu')(x)
    flat = layers.Flatten()(x)
    outputs=layers.Dense(10,activation='softmax')(flat)
    model = keras.Model(inputs, outputs)
    model.summary()
    model_ob=keras.Model(inputs, flat)
    #train----
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history=model.fit(x=x_train_normal,  
                          y=y_train_onehot, validation_split=0.2,
                          epochs=20, batch_size=1024, verbose=1)
    #evaluate
    print(model.evaluate(x_test_normal,tf.squeeze(y_text_onehot),batch_size=1024))
    print_data=model_ob.predict(x_train_normal,batch_size=256)
    embedding_plot(print_data,y_train)
    show_train_history(history,'accuracy','val_accuracy')
    #tf.keras.utils.plot_model(model,show_shapes=True)