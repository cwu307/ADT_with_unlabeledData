'''
This script is to test the trained encoder
CW @ GTCMT 2017
'''

import numpy as np
from keras.models import load_model
from librosa.features import melspectrogram
from ../autoencoder/FileUtil import reshapeInputTensor, convert2dB, scaleTensorTrackwise
from FileUtil import averageActivationMap, standardizeTensorTrackwise, normalizeTensorTrackwiseL1, convert2dB
preprocessingFlag = True


def extractConvFeatures(inputMatrix, modelSavePath):
    ae_path = modelSavePath + 'ae.h5'
    ext1Path = modelSavePath + 'ext1.h5'
    ext2Path = modelSavePath + 'ext2.h5'
    ext3Path = modelSavePath + 'ext3.h5'
    ext4Path = modelSavePath + 'ext4.h5'
    ext5Path = modelSavePath + 'ext5.h5'

    ext5_model = load_model(ext5_path)
    numFreq, numBlock = np.shape(inputMatrix)
    inputTensor = prepareConvnetInput(inputMatrix)
    lay5Out = ext5_model.predict(X_train, batch_size=1)  # N x 8 x 8 x 128
    lay5OutFlat = flattenConvFeatures(lay5Out)
    numFeat, numBlock2  = np.shape(lay5OutFlat)
    #zero-padding here
    if numBlock > numBlock2:
        zpad = np.zeros((numFeat, numBlock - numBlock2))
        convFeatures = np.concatenate((lay5Flat, zpad), axis=1)
    else:
        convFeatures = lay5Flat[:, 0:numBlock]
    return convFeatures

'''
layerOutFlat = flattenConvFeatures(layerOut)
input: 
    layerOut: numSample x numChannel x dim1 x dim2 (e.g., 19 x 8 x 8 x 128)
output:
    layerOutFlat: numSample2 x numFeatures (e.g., 64 x 2432)
'''
def flattenConvFeatures(layerOut):
    numSample = np.size(layerOut, axis=0)
    numChannel = np.size(layerOut, axis=1)
    layerOutFlat = []
    for i in range(0, numSample):
        tmp1 = []
        for j in range(0, numChannel):
            cur = layerOut[i, j, :, :]
            if len(tmp1) == 0:
                tmp1 = cur
            else:  
                tmp1 = np.concatenate((tmp1, cur), axis=0)
        #combine rectangular feature maps
        if len(layerOutFlat) == 0:
            layerOutFlat = tmp1
        else:  
            layerOutFlat = np.concatenate((layerOutFlat, tmp1), axis=1)
    return layerOutFlat

'''
inputTensor = prepareConvnetInput(inputMatrix)
input:
    inputMatrix: magnitude spectrogram, numFreq x numBlock
output:
    inputTensor: numBatch x numChannel x dim1 x dim2
'''
def prepareConvnetInput(inputMatrix):    
    inputMatrixMel = melspectrogram(S=inputMatrix, sr=44100, n_fft=2048, hop_length=512, power=2.0, n_mels=128, fmin=0.0, fmax=20000)
    inputTensor = np.expand_dims(inputMatrixMel, axis=0)
    if preprocessingFlag:
        inputTensor = convert2dB(inputTensor)
    inputTensor = scaleTensorTrackwise(inputTensor)
    inputTensor = reshapeInputTensor(inputTensor)
    inputTensor = np.expand_dims(inputTensor, axis=1) #add batch dimension 1 x 1 x dim1 x dim2
    return inputTensor

def main():
    modelSavePath = './savedAeModels/'
    return()

if __name__ == "__main__":
    print('running main() directly')
    main()
