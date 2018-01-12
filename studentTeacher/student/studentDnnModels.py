'''
this script is a collection of all basic student DNN models for testing
CW @ GTCMT 2017
'''
import numpy as np
import keras.backend as K
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Conv2DTranspose, GlobalAveragePooling2D, UpSampling2D, BatchNormalization, Flatten, Dropout
from keras.models import Model
from keras.initializers import he_normal, RandomNormal, RandomUniform

'''
Testing similar architecture as described in Keunwoochoi's paper
'''
def createCnnModel(inputDim, inputDim2, selectedOptimizer, selectedLoss):
    #fixed_init = RandomNormal(mean=0.0, stddev=0.5, seed=10)
    fixed_init = he_normal(seed=0)
    #print('autoencoder model')
    input = Input(shape=(1, inputDim, inputDim2)) #1 x 128 x 128
    out1 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(input) #32 x 128 x 128
    out1 = BatchNormalization(axis=1)(out1)
    out1 = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(out1)  #32 x 64 x 64
    out2 = Convolution2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out1) #16 x 64 x 64
    out2 = BatchNormalization(axis=1)(out2)
    out2 = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(out2) #16 x 32 x 32
    out3 = Convolution2D(8, (3, 3), activation='relu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out2) #8 x 32 x 32
    out3 = BatchNormalization(axis=1)(out3)
    out3 = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(out3)  #8 x 16 x 16
    out4 = Convolution2D(4, (3, 3), activation='relu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out3) #4 x 16 x 16
    out4 = BatchNormalization(axis=1)(out4)
    out4 = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(out4)  #4 x 8 x 8
    out5 = Flatten()(out4) #(256)
    out6 = Dense(units=32, activation='relu')(out5)
    out7 = Dense(units=3, activation='sigmoid')(out6)

    #==== create model
    cnnModel = Model(input, out7)

    #==== compile model
    cnnModel.compile(optimizer=selectedOptimizer, loss=selectedLoss, metrics=['mae'])
    return cnnModel


def createFcModel(inputDim, selectedOptimizer, selectedLoss):
    input = Input(shape=(inputDim,))
    out1 = Dense(units=1025, activation='relu')(input)
    out2 = Dense(units=512, activation='relu')(out1)
    out3 = Dense(units=32, activation='relu')(out2)
    out4 = Dense(units=3, activation='sigmoid')(out3)

    fcModel = Model(input, out4)
    fcModel.compile(optimizer=selectedOptimizer, loss=selectedLoss, metrics=['mae'])
    return fcModel