'''
this script uses the pre-trained STUDENT/TEACHER models to transcribe drum events in the held-out dataset
the script is similar to ../featureLearning/runPrediction.py with some modifications
CW @ GTCMT 2018
'''
import numpy as np
import sys
import os
import time
sys.path.insert(0, './student')
sys.path.insert(0, './teacher')
sys.path.insert(0, '../featureLearning/')
from runTeachers import useAdtlibSingleFile
from transcriptUtil import medianThreshold, thresNvt, findPeaks
from librosa.core import load, stft
from pfNmf import pfNmf
HOPSIZE = 512
WINSIZE = 2048
FS = 44100.0
ORDER = round(0.1 / (HOPSIZE / FS))
OFFSET = 0.12

'''
input:
    heldOutOption: str, viable options are 'enst', 'mdb', 'rbma', 'm2005'
    modelOption: str, viable options are 'FC', 'pfnmf_200d', 'pfnmf_smt'
output:
    None
    the predictions are saved as .txt files and stored in ./predictionResults/ folder
'''
def predictHeldOutDataset(heldOutOption, modelOption):
    dataListPath = '../featureLearning/featureExtraction/dataLists/' + heldOutOption + 'List.npy'
    modelFolder = './student/savedStudentModels/'
    
    saveFolder = './predictionResults/' + heldOutOption + '_feat_' + modelOption + '/'
    if not os.path.isdir(saveFolder):
        os.makedirs(saveFolder)
    
    dataList = np.load(dataListPath)
    predictionListPath = saveFolder + heldOutOption + '_feat_' + modelOption + '_predictionList.npy'
    predictionList = []
    c = 0
    for audioPath, annPath in dataList:
        print('processing %s ...' % audioPath)
        c += 1 #assign each prediction file a unique number
        predictions = predictOneSongActivBased(audioPath, modelOption)
        predictionSavePath = getPredictionSavePath(annPath, saveFolder, 'txt', count=c)
        writeResults2File(predictions, predictionSavePath)
        #==== create another datalist for ease of use
        predictionList.append((annPath, predictionSavePath))
    np.save(predictionListPath, predictionList)
    return()

def getPredictionSavePath(annPath, saveFolder, ext, count):
    fileName = annPath.split('/')[-1][0:-3]
    predictionSavePath = saveFolder + str(count) + '_' + fileName + ext
    return predictionSavePath

def writeResults2File(predictions, predictionSavePath):
    txtFile = open(predictionSavePath, 'w')
    for time, drum in predictions:
        line = str(time) + '    ' + drum + '\n'
        txtFile.write(line)
    txtFile.close()
    return ()

'''
note:
    the order of the drums--> 0:KD, 1:SD, 2:HH
'''
def getActivationFromFile(audioPath, modelOption):
    y, sr = load(audioPath, sr=FS, mono=True)
    y = np.divide(y, max(abs(y)))
    S = stft(y, n_fft=WINSIZE, hop_length=HOPSIZE, window='hann')
    inputMatrix = abs(S)

    if modelOption == 'pfnmf_200d':
        WD = np.load('./teacher/drumTemplates/template_200drums_2048_512.npy')
        WD, HD, WH, HH, err = pfNmf(inputMatrix, WD, HD=[], WH=[], HH=[], rh=50, sparsity=0.0)
        activMat = np.transpose(HD)
    elif modelOption == 'pfnmf_smt':
        WD = np.load('./teacher/drumTemplates/template_smt_2048_512.npy')
        WD, HD, WH, HH, err = pfNmf(inputMatrix, WD, HD=[], WH=[], HH=[], rh=50, sparsity=0.0)
        activMat = np.transpose(HD)
    elif modelOption == 'FCRandom':
        from keras.models import load_model
        modelPath = './student/savedStudentModels/' + modelOption + '/studentModel.h5'
        studentModel = load_model(modelPath)
        inputMatrix = np.transpose(inputMatrix)
        activMat = studentModel.predict(inputMatrix)
    elif modelOption == 'FC':
        from keras.models import load_model
        modelPath = './student/savedStudentModels/' + modelOption + '/studentModel.h5'
        studentModel = load_model(modelPath)
        inputMatrix = np.transpose(inputMatrix)
        activMat = studentModel.predict(inputMatrix)
    elif modelOption == 'adtlib':
        numFreq, numBlock = np.shape(inputMatrix)
        activMat = useAdtlibSingleFile(audioPath, numBlock) 
        activMat = np.transpose(activMat) #the dimensionality should be numBlock x numDrums
    return activMat


'''
given the path to audio file, extract the selected feature and transcribe the drum events
using activation-based method (extract activation functions and then detect onsets)
input
    audioPath: str, the path to the target audio file
    modelOption: str, viable options are 'FC', 'FCRandom', 'pfnmf_200d', 'pfnmf_smt'
output
    predictions: list of tuples (onsetTimeInSec, drumTypeInStr)
'''
def predictOneSongActivBased(audioPath, modelOption):
    predictions = []
    activMat = getActivationFromFile(audioPath, modelOption)
    activBd = activMat[:, 0]
    activSd = activMat[:, 1]
    activHh = activMat[:, 2]

    nvtBd = np.divide(activBd, np.max(activBd))
    nvtSd = np.divide(activSd, np.max(activSd))
    nvtHh = np.divide(activHh, np.max(activHh))

    thresCurvBd = medianThreshold(nvtBd, ORDER, OFFSET)
    thresCurvSd = medianThreshold(nvtSd, ORDER, OFFSET)
    thresCurvHh = medianThreshold(nvtHh, ORDER, OFFSET)

    dump, onsetsInSecBd = findPeaks(nvtBd, thresCurvBd, FS, HOPSIZE)
    dump, onsetsInSecSd = findPeaks(nvtSd, thresCurvSd, FS, HOPSIZE)
    dump, onsetsInSecHh = findPeaks(nvtHh, thresCurvHh, FS, HOPSIZE)

    allOnsetLists = [onsetsInSecBd, onsetsInSecSd, onsetsInSecHh]
    predictions = []
    for i in range(0, len(allOnsetLists)):
        for onsetInSec in allOnsetLists[i]:
            if i == 0:
                predictions.append((onsetInSec, 'KD'))
            elif i == 1:
                predictions.append((onsetInSec, 'SD'))
            elif i == 2:
                predictions.append((onsetInSec, 'HH'))
    predictions = sorted(predictions)
    return predictions

def main():
    allModelOptions = ['adtlib']#['pfnmf_200d', 'pfnmf_smt']#['FC', 'FCRandom', 'pfnmf_200d', 'pfnmf_smt'  ]
    allHeldOutOptions = ['enst', 'mdb', 'rbma', 'm2005']
    
    for modelOption in allModelOptions:
        for heldOutOption in allHeldOutOptions:
            print('current model = %s, current dataset = %s' % (modelOption, heldOutOption))
            predictHeldOutDataset(heldOutOption, modelOption)
    return()

if __name__=="__main__":
    print('running main() directly')
    main()



