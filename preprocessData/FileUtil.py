# A collection of some of the utility functions
# CW @ GTCMT 2017

from os import listdir
import numpy as np

'''
input:
    parentFolder: string, directory to the parent folder
    ext: string, extension name of the interested files
output:
    filePathList: list, directory to the files
'''
def getFilePathList(folderpath, ext):
    allfiles = listdir(folderpath)
    print(listdir(folderpath))
    quit()
    filePathList = []
    filenames = []
    for filename in allfiles:
        if ext in filename:
            filepath = folderpath + filename
            filePathList.append(filepath)
            filenames.append(filename)
    return filePathList, filenames

'''
input:
    vector: N by 1 float vector
output:
    vector_scaled: N by 1 float vector max = 1 min = 0
    maxValue: max value in original data vector
    minValue: min value in original data vector
'''
def scaleVector(vector):
    maxValue = max(vector)
    minValue = min(vector)
    vector_scaled = np.divide(np.subtract(vector, minValue), float(np.subtract(maxValue, minValue)))
    return vector_scaled, maxValue, minValue


'''
input:
    vector: N by 1 float vector
    maxValue: max value in original data vector
    minValue: min value in original data vector
output:
    vector_scaled: N by 1 float vector max = 1 min = 0
'''
def scaleVectorWithMinMax(vector, maxValue, minValue):
    vector_scaled = np.divide(np.subtract(vector, minValue), float(np.subtract(maxValue, minValue)))
    return vector_scaled


'''
input:
    matrix: N by M float matrix, N is the #blocks, M is the #freq bins
output:
    matrix_scaled: N by M float matrix
    maxVector: 1 by M float vector of max values for each bin
    minVector: 1 by M float vector of min values for each bin
'''
def scaleMatrix(matrix):
    maxVector = np.max(matrix, axis=0)
    minVector = np.min(matrix, axis=0)
    matrix_scaled = np.divide(np.subtract(matrix, minVector), np.subtract(maxVector, minVector))
    return matrix_scaled, maxVector, minVector

'''
input:
    matrix: N by M float matrix, N is the #blocks, M is the #freq bins
    maxVector: 1 by M float vector of max values for each bin
    minVector: 1 by M float vector of min values for each bin
output:
    matrix_scaled: N by M float matrix
'''
def scaleMatrixWithMinMax(matrix, maxVector, minVector):
    matrix_scaled = np.divide(np.subtract(matrix, minVector), np.subtract(maxVector, minVector))
    return matrix_scaled


'''
input:
    data_tensor: numSample x numFreq x numBlock
output:
    data_tensor_reshaped: numSample_mod x numFreq x numFreq
e.g., the original is N x M1 x M2
the resulting would be N' x M1 X M1
'''
def reshapeInputTensor(data_tensor):
    data_tensor_reshaped = []
    numSample = np.size(data_tensor, 0)
    numFreq   = np.size(data_tensor, 1)
    numBlock  = np.size(data_tensor, 2)

    numSubBlock = int(np.floor(np.divide(numBlock, numFreq)))
    numBlock_mod = int(np.multiply(numSubBlock, numFreq))
    data_tensor  = data_tensor[:, :, 0:numBlock_mod]
    for i in range(0, numSample):
        print(i)
        cur = data_tensor[i, :, :]
        tmp = np.reshape(cur, (numSubBlock, numFreq, numFreq))
        if len(data_tensor_reshaped) == 0:
            data_tensor_reshaped = tmp
        else:
            data_tensor_reshaped = np.concatenate((data_tensor_reshaped, tmp), axis=0)

    print(np.shape(data_tensor))
    print(np.shape(data_tensor_reshaped))
    return data_tensor_reshaped

'''
Average the activation map across both time and freq
input: 
    data_tensor: tensor, numSample x numFreq x numBlock
output:
    feature_mat: array, numSample x ch
'''
def averageActivationMap(data_tensor):
    tmp = np.mean(data_tensor, axis=2)
    feature_mat = np.mean(tmp, axis=2)
    return feature_mat


'''
Track-wise z-score standardization on tensor
input:
    data_tensor: tensor, numSample x numFreq x numBlock
output:
    data_tensor: normalized tensor, same dimensionality as input
'''
def standardizeTensorTrackwise(data_tensor):
    numSample, numFreq, numBlock = np.shape(data_tensor)
    for i in range(0, numSample):
        avg = np.mean(data_tensor[i, :, :])
        std = np.std(data_tensor[i, :, :])
        if std != 0:
            data_tensor[i, :, :] = np.divide(data_tensor[i, :, :] - avg + 10e-6, std)
        else:
            data_tensor[i, :, :] = 0 * data_tensor[i, :, :]
    return data_tensor

