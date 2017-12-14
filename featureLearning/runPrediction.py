'''
this script uses the pre-trained classifiers to transcribe drum events in the held-out dataset
CW @ GTCMT 2017
'''
import numpy as np
import sys
import os
import time
sys.path.insert(0, './autoencoder')
sys.path.insert(0, './mainTaskModels')
sys.path.insert(0, './featureExtraction')
from FileUtil import scaleMatrixWithMinMax
from buildFeatMatrix import parseAnnotations, featureSplicing
from trainClassifier import getClassifierPath
from extractFeatures import extractRandomConvFeatures, extractBaselineFeatures, extractConvFeatures
from madmom.features.onsets import CNNOnsetProcessor, RNNOnsetProcessor
from madmom.features.onsets import OnsetPeakPickingProcessor

HOPSIZE = 512
FS = 44100.0

'''
input:
    heldOutOption: str, viable options are 'enst', 'mdb', 'rbma', 'm2005'
    featureOption: str, viable options are 'baseline', 'convAe', 'convDae', 'convRandom'
output:
    None
    the predictions are saved as .txt files and stored in ./predictionResults/ folder
'''
def predictHeldOutDataset(heldOutOption, featureOption):
    dataListPath = './featureExtraction/dataLists/' + heldOutOption + 'List.npy'
    modelFolder = './mainTaskModels/trainedClassifier/'
    clfModelPath = getClassifierPath(heldOutOption, featureOption, modelFolder)

    saveFolder = './predictionResults/' + heldOutOption + '_feat_' + featureOption + '/'
    if not os.path.isdir(saveFolder):
        os.makedirs(saveFolder)
    
    dataList = np.load(dataListPath)
    predictionListPath = saveFolder + heldOutOption + '_feat_' + featureOption + '_predictionList.npy'
    predictionList = []
    c = 0
    for audioPath, annPath in dataList:
        print('processing %s ...' % audioPath)
        c += 1 #assign each prediction file a unique number
        predictions = predictOneSong(audioPath, featureOption, clfModelPath)
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
given the path to audio file, extract the selected feature and transcribe the drum events
input
    audioPath: str, the path to the target audio file
    featureOption:
    clfModelPath:
output
    predictions: list of tuples (onsetTimeInSec, drumTypeInStr)
'''
def predictOneSong(audioPath, featureOption, clfModelPath):
    predictions = []
    if featureOption == 'convRandom':
        features = extractRandomConvFeatures(audioPath) #64 x M            
    elif featureOption == 'convAe':
        modelSavePath = '../autoencoder/savedAeModels/'
        features = extractConvFeatures(audioPath, modelSavePath)
    elif featureOption == 'convDae':
        modelSavePath = '../autoencoder/savedDaeModels/'
        features = extractConvFeatures(audioPath, modelSavePath)
    elif featureOption == 'baseline':
        features = extractBaselineFeatures(audioPath) #60 x M
    else:
        print('unknown feature option')
    #==== onset detection
    onsetDetector = CNNOnsetProcessor()
    nvt = onsetDetector(audioPath)
    peakPicker = OnsetPeakPickingProcessor(fps=100)
    onsets = peakPicker(nvt)
    onsetsInFrames = [round(np.divide(onset, HOPSIZE/FS)) for onset in onsets]

    #==== collect feature of interest
    X = []
    timeStamp = []
    for i in range(0, len(onsetsInFrames)):
        midIndex = int(onsetsInFrames[i])
        curTime  = onsets[i]
        splicedFeature = featureSplicing(features, midIndex, 1, 2)
        X.append(splicedFeature)
        timeStamp.append(curTime)
    #print(np.shape(X))
    #==== drum transcription
    tmp = np.load(clfModelPath)
    classifiers = tmp['arr_0']
    normParams = tmp['arr_1']
    clfBd = classifiers[0]
    clfSd = classifiers[1]
    clfHh = classifiers[2]
    maxVec = normParams[0]
    minVec = normParams[1]
    XScaled = scaleMatrixWithMinMax(X, maxVec, minVec)

    predictions = []
    for i in range(0, len(timeStamp)):
        curFeature = X[i]
        curFeature = np.expand_dims(curFeature, axis=0)
        detectBd = clfBd.predict(curFeature)
        detectSd = clfSd.predict(curFeature)
        detectHh = clfHh.predict(curFeature)

        if detectBd:
            predictions.append((timeStamp[i], 'KD'))
        if detectSd:
            predictions.append((timeStamp[i], 'SD'))
        if detectHh:
            predictions.append((timeStamp[i], 'HH'))
    return predictions

def main():
    allFeatureOptions = ['baseline', 'convRandom']#, 'convAe', 'convDae']
    allHeldOutOptions = ['enst', 'mdb', 'rbma', 'm2005']
    
    for featureOption in allFeatureOptions:
        for heldOutOption in allHeldOutOptions:
            predictHeldOutDataset(heldOutOption, featureOption)
    return()

if __name__=="__main__":
    print('running main() directly')
    main()



