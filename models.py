import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
from layers import *

class h_extrema_denoising_block2(tf.keras.Model):
    def __init__(self, dropout=0.2, name="h_extrema_denoising_block2"):       
        super(h_extrema_denoising_block2, self).__init__(name=name)
        self.conv1 = kl.Conv2D(filters = 8, kernel_size = (3, 3), activation='relu',kernel_initializer='glorot_uniform')
        self.conv2 = kl.Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',kernel_initializer='glorot_uniform')
        self.conv3 = kl.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu',kernel_initializer='glorot_uniform')
        self.conv4 = kl.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu',kernel_initializer='glorot_uniform')
        self.maxpooling = kl.MaxPool2D(pool_size=(2,2), padding="valid")
        self.maxpooling2 = kl.MaxPool2D(pool_size=(2,2), padding="valid")
        self.dropout = kl.Dropout(dropout)
        self.globalavgpooling = kl.GlobalMaxPooling2D(data_format='channels_last')
        self.dense = kl.Dense(1,kernel_constraint=tf.keras.constraints.NonNeg(),name="h_denoising")
        self.batchnorm1 = kl.BatchNormalization()
        self.batchnorm2 = kl.BatchNormalization()
        self.batchnorm3 = kl.BatchNormalization()
        self.batchnorm4 = kl.BatchNormalization()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.maxpooling(x)        
        x = self.batchnorm2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.conv4(x)
        x = self.maxpooling2(x)
        x = self.globalavgpooling(x)        
        x = self.dense(x)
        h = x
        return h


class H_maxima_model:
    def __init__(self, input_shape, back_prop_mode='minus_one', N=50, count_only=True):
        self.input_shape = input_shape
        self.N = N
        self.bp_mode=back_prop_mode
        self.count_only=count_only
        self.h_extrema_denoising_block_ = h_extrema_denoising_block2()
        self.nn, self.nn_h = self.get_simple_model()

    def get_simple_model(self):
        xin=kl.Input(shape=self.input_shape)
        xinput=xin
        h=self.h_extrema_denoising_block_(xinput)
        bp_mode = self.bp_mode
        if self.count_only:
            NCC = countingLayer(bp_mode = bp_mode, N=self.N)([xinput,h])
            return tf.keras.Model(xin, NCC), tf.keras.Model(xin,h)
        else:
            NCC, Xrec = countAndRecLayer(bp_mode = bp_mode, N=self.N)([xinput,h])
            NCC = ExpandtoImageLayer()(NCC)
            NCC = kl.Conv2D(filters = 1, kernel_size = (1, 1), kernel_initializer='Ones', name='CountOutput', trainable=False)(NCC)
            Xrec = kl.Conv2D(filters = 1, kernel_size = (1, 1), kernel_initializer='Ones', name='RecOutput', trainable=False)(Xrec)
            return tf.keras.Model(xin,[NCC, Xrec]), tf.keras.Model(xin,h)
    






    

