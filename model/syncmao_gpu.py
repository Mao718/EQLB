import tensorflow as tf
import numpy as np
import time


class SyncMap:
    def __init__(self, input_size: int, dimension: int, init_w: 'nparray' = None, lr_rate=0.01):
        self.input_size = input_size
        self.dimension = dimension
        self.lr_rate = lr_rate
        if init_w is not None:
            self.init_w = init_w
            if init_w.shape != (input_size, dimension):
                print('init error')
                return
        else:
            # check
            self.init_w = np.random.rand(input_size, dimension)
        print(self.init_w)

    def input(self, input_sequence: 'nparray(batch,timestep,inputsize)'):
        syncmap = tf.expand_dims(tf.Variable(self.init_w.copy()), 0)
        syncmap = tf.tile(syncmap, tf.constant(
            [input_sequence.shape[0], 1, 1]))
        self.output = tf.vectorized_map(
            lambda x: self.inputGeneral(x[0], x[1]), (input_sequence, syncmap))
        return self.output

    def inputGeneral(self, x, syncmap):

        plus = tf.where(x > 0.1, 1, 0)
        minus = tf.where(x > 0.1, 0, 1)
        # for t in range(x.shape[0]):
        #vplus = plus[t, :]
        #vminus = minus[t, :]


if __name__ == "__main__":

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    start = time.time()
    test_x = np.random.rand(20, 1000, 10)
    model = SyncMap(10, 3)
    out = model.input(test_x)
    end = time.time()
    print("using time ", start-end)
    # print(out)
