'''
This script is to test the trained encoder
CW @ GTCMT 2017
'''

import numpy as np
import sys
sys.path.insert(0, '../autoencoder')

from keras.models import load_model
from keras.optimizers import Adam
from librosa.feature import melspectrogram, mfcc
from librosa.core import load, stft
from FileUtil import reshapeInputTensor, convert2dB, scaleTensorTrackwise
from dnnModels import createAeModel
from FileUtil import averageActivationMap, standardizeTensorTrackwise, normalizeTensorTrackwiseL1, convert2dB


'''
convFeatures = extractConvFeatures(filePath, modelSavePath)
input:
    filePath: str, path to the target audio file
    modelSavePath: str, path where the CNN model is stored
output:
    convFeatures: ndarray, numFeatures x numBlock
'''
def extractConvFeatures(filePath, modelSavePath):
    y, sr = load(filePath, sr=44100, mono=True)
    y = np.divide(y, max(abs(y)))
    S = stft(y, n_fft=2048, hop_length=512, window='hann')
    inputMatrix = abs(S)
    ext5Path = modelSavePath + 'ext5.h5'
    ext5_model = load_model(ext5Path)
    numFreq, numBlock = np.shape(inputMatrix)
    inputTensor = prepareConvnetInput(inputMatrix)
    lay5Out = ext5_model.predict(inputTensor, batch_size=1)  # N x 8 x 8 x 128
    lay5OutFlat = flattenConvFeatures(lay5Out)
    numFeat, numBlock2  = np.shape(lay5OutFlat)
    #zero-padding here
    if numBlock > numBlock2:
        zpad = np.zeros((numFeat, numBlock - numBlock2))
        convFeatures = np.concatenate((lay5OutFlat, zpad), axis=1)
    else:
        convFeatures = lay5OutFlat[:, 0:numBlock]
    return convFeatures

'''
convFeatures = extractRandomConvFeatures(filePath)
input:
    filePath: str, path to the target audio file
output:
    convFeatures: ndarray, numFeatures x numBlock
'''
def extractRandomConvFeatures(filePath, modelSavePath):
    y, sr = load(filePath, sr=44100, mono=True)
    y = np.divide(y, max(abs(y)))
    S = stft(y, n_fft=2048, hop_length=512, window='hann')
    inputMatrix = abs(S)
    ext5Path = modelSavePath + 'ext5.h5'
    ext5_model = load_model(ext5Path)
    numFreq, numBlock = np.shape(inputMatrix)
    inputTensor = prepareConvnetInput(inputMatrix)
    lay5Out = ext5_model.predict(inputTensor, batch_size=1)  # N x 8 x 8 x 128
    lay5OutFlat = flattenConvFeatures(lay5Out)
    numFeat, numBlock2  = np.shape(lay5OutFlat)
    #zero-padding here
    if numBlock > numBlock2:
        zpad = np.zeros((numFeat, numBlock - numBlock2))
        convFeatures = np.concatenate((lay5OutFlat, zpad), axis=1)
    else:
        convFeatures = lay5OutFlat[:, 0:numBlock]
    return convFeatures

def checkNan(inputTensor):
    nanCount = 0
    for element in np.nditer(inputTensor):
        if np.isnan(element):
            nanCount += 1

    if nanCount > 0:
        return True
    else:
        return False

'''
layerOutFlat = flattenConvFeatures(layerOut)
input: 
    layerOut: numSample x numChannel x dim1 x dim2 (e.g., 19 x 8 x 8 x 128)
output:
    layerOutFlat: numSample2 x numFeatures (e.g., 8 x 2432)
'''
def flattenConvFeatures(layerOut):
    numSample = np.size(layerOut, axis=0)
    numChannel = np.size(layerOut, axis=1)
    layerOutFlat = []
    for i in range(0, numSample):
        tmp1 = []
        for j in range(0, numChannel):
            cur = layerOut[i, j, :, :]
            cur = np.mean(cur, axis=0)
            cur = np.expand_dims(cur, axis=0)
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
    inputTensor: numSeg x numChannel x dim1 x dim2 
    dim1 = n_mels
    dim2 = dim1
'''
def prepareConvnetInput(inputMatrix):    
    inputMatrixMel = melspectrogram(S=inputMatrix, sr=44100, n_fft=2048, hop_length=512, power=2.0, n_mels=128, fmin=0.0, fmax=20000)
    inputTensor = np.expand_dims(inputMatrixMel, axis=0) #add a dummy dimension for sample count
    inputTensor = convert2dB(inputTensor) #the input is the power of mel spectrogram
    inputTensor = scaleTensorTrackwise(inputTensor) #scale the dB scaled tensor to range of {0, 1}
    inputTensor = reshapeInputTensor(inputTensor) #break one long matrix into stacked segments
    inputTensor = np.expand_dims(inputTensor, axis=1) #add channel dimension 1 x 1 x dim1 x dim2
    return inputTensor


'''
baselineFeatures = extractBaselineFeatures(filePath)
input:
    filePath: str, path to the target audio file
output:
    baselineFeatures: ndarray, numFeatures x numBlock
                      the features are the concatenation of 
                      [20 MFCCs, 
                       20 delta MFCCs
                       20 delta delta MFCCs]
'''
def extractBaselineFeatures(filePath):
    y, sr = load(filePath, sr=44100, mono=True)
    y = np.divide(y, max(abs(y)))    
    mfccs = mfcc(y=y, sr=sr, n_mfcc=20)
    dMfccs = np.zeros(np.shape(mfccs))
    dMfccs[:, 0:-1] = np.diff(mfccs, axis=1)
    ddMfccs = np.zeros(np.shape(mfccs))
    ddMfccs[:, 0:-1] = np.diff(dMfccs, axis=1)
    baselineFeatures = np.concatenate((mfccs, dMfccs, ddMfccs), axis=0)
    return baselineFeatures

def main(): 
    modelSavePath = '../autoencoder/savedAeModels/'
    filePath = '/data/unlabeledDrumDataset/audio/alternative-songs/1_WhenYourHeartStopsBeating44.mp3'
    convFeatures = extractRandomConvFeatures(filePath)
    print('A quick test of one feature extractor')
    print('file = %s' % filePath)
    print(np.shape(convFeatures))
    return()

if __name__ == "__main__":
    print('running main() directly')
    main()
