'''
This script is to extract train a basic auto-encoder for feature learning project
Note:
    1) Sequentially train the AE (file per file)
    2) Convolutional AE
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
def createAeModel(inputDim, inputDim2, embedDim, selectedOptimizer, selectedLoss):
    #fixed_init = RandomNormal(mean=0.0, stddev=0.5, seed=10)
    fixed_init = he_normal(seed=0)
    #print('autoencoder model')
    input = Input(shape=(1, inputDim, inputDim2)) #1 x 128 x 128
    out1 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(input) #32 x 128 x 128
    out1 = BatchNormalization(axis=1)(out1)
    out1 = MaxPooling2D((2, 1), padding='same', data_format='channels_first')(out1)  #32 x 64 x 128
    out2 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out1) #32 x 64 x 128
    out2 = BatchNormalization(axis=1)(out2)
    out2 = MaxPooling2D((2, 1), padding='same', data_format='channels_first')(out2) #32 x 32 x 128
    out3 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out2) #16 x 32 x 128
    out3 = BatchNormalization(axis=1)(out3)
    out3 = MaxPooling2D((2, 1), padding='same', data_format='channels_first')(out3)  #16 x 16 x 128
    out4 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out3) #16 x 16 x 128
    out4 = BatchNormalization(axis=1)(out4)
    out4 = MaxPooling2D((2, 1), padding='same', data_format='channels_first')(out4)  #16 x 8 x 128

    encoded = Convolution2D(embedDim, (3, 3), activation='relu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out4) #embedDim x 8 x 128
    
    out5 = UpSampling2D((2, 1), data_format='channels_first')(encoded)  
    out5 = Convolution2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out5)  
    out5 = BatchNormalization(axis=1)(out5)
    out6 = UpSampling2D((2, 1), data_format='channels_first')(out5)  
    out6 = Convolution2D(16, (3, 3), activation='relu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out6)  
    out6 = BatchNormalization(axis=1)(out6)
    out7 = UpSampling2D((2, 1), data_format='channels_first')(out6)  
    out7 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out7)
    out7 = BatchNormalization(axis=1)(out7)
    out8 = UpSampling2D((2, 1), data_format='channels_first')(out7)  
    out8 = Convolution2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out8)  
    out8 = BatchNormalization(axis=1)(out8)
    output = Convolution2D(1, (1, 1), activation='sigmoid', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out8) 

    #==== create model
    autoencoder = Model(input, output)
    layer1Extractor = Model(input, out1)
    layer2Extractor = Model(input, out2)
    layer3Extractor = Model(input, out3)
    layer4Extractor = Model(input, out4)
    bottleneckExtractor = Model(input, encoded)

    #==== compile model
    autoencoder.compile(optimizer=selectedOptimizer, loss=selectedLoss, metrics=['mae'])
    return autoencoder, layer1Extractor, layer2Extractor, layer3Extractor, layer4Extractor, bottleneckExtractor

'''
Testing similar architecture as described in Keunwoochoi's paper
'''
def createModel_cqt_classification_fma_medium(input_dim, input_dim2, selected_optimizer, selected_loss):
    fixed_init = he_normal(seed=0)
    print('classifier model')
    input = Input(shape=(1, input_dim, input_dim2)) #1 x 80 x 1280 
    out1 = Convolution2D(32, (3, 3), activation='elu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(input) #32 x 80 x 1280
    out1 = Dropout(rate=0.1)(out1)
    out1 = BatchNormalization(axis=1)(out1)
    out1 = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(out1)  #32 x 40 x 640
    out2 = Convolution2D(32, (3, 3), activation='elu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out1) #32 x 40 x 640
    out2 = Dropout(rate=0.1)(out2)
    out2 = BatchNormalization(axis=1)(out2)
    out2 = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(out2) #32 x 20 x 320
    out3 = Convolution2D(32, (3, 3), activation='elu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out2) #32 x 20 x 320
    out3 = Dropout(rate=0.1)(out3)
    out3 = BatchNormalization(axis=1)(out3)
    out3 = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(out3)  #32 x 10 x 160
    out4 = Convolution2D(32, (3, 3), activation='elu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out3) #32 x 10 x 160
    out4 = Dropout(rate=0.1)(out4)
    out4 = BatchNormalization(axis=1)(out4)
    out4 = MaxPooling2D((2, 2), padding='same', data_format='channels_first')(out4)  #32 x 5 x 80

    out5 = Convolution2D(32, (3, 3), activation='elu', padding='same', data_format='channels_first', kernel_initializer=fixed_init)(out4) #32 x 5 x 80
    out5 = Dropout(rate=0.1)(out5)
    out5 = BatchNormalization(axis=1)(out5)
    out5 = MaxPooling2D((5, 5), padding='same', data_format='channels_first')(out5)  #32 x 1 x 16
    
    out6 = GlobalAveragePooling2D(data_format='channels_first')(out5) #same as previous
    output = Dense(16, activation='softmax')(out6)

    #==== create model
    classifier = Model(input, output)
    layer1_extractor = Model(input, out1)
    layer2_extractor = Model(input, out2)
    layer3_extractor = Model(input, out3)
    layer4_extractor = Model(input, out4)
    layer5_extractor = Model(input, out5)
    #layer6_extractor = Model(input, out4i)

    #==== compile model
    classifier.compile(optimizer=selected_optimizer, loss=selected_loss, metrics=['acc'])
    return classifier, layer1_extractor, layer2_extractor, layer3_extractor, layer4_extractor, layer5_extractor


