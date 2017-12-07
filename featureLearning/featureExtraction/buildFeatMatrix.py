'''
this script is used to build feature matrices
CW @ GTCMT 2017
'''
import numpy as np
import os
from extractFeatures import extractRandomConvFeatures, extractBaselineFeatures, extractConvFeatures
HOPSIZE = 512
FS = 44100.0

'''
input
    listPath: dataList genereated using "prepareDataList.py". (e.g., './dataLists/enstList.npy')
                dataList is an array of tuples: (audioPath, annotationPath)
    savePath: str, the location to save the feature matrix (e.g., './featureMat/enstFeatureMat.npy')
                the saved .npy file contains an array of [X, y, originalFilePath]
    featureOption: str, available options are 'convRandom', 'convAe', 'convDae', 'baseline' 
output
    None
'''
def getFeatureMatrix(listPath, savePath, featureOption):
    dataList = np.load(listPath)
    numFiles = len(dataList)
    saveFolder = './featureMat/'
    if not os.path.isdir(saveFolder):
        os.mkdir(saveFolder) 
    originalFilePath = []
    X = []
    y = []
    
    for i in range(0, numFiles):
        audioPath, annPath = dataList[i]
        print('Processing file %d ...' % i)
        print(audioPath)
        print(annPath)
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

        onsetInFrames, classInNum = parseAnnotations(annPath)
        neighbor2Include = 1
        for j in range(0, len(classInNum)):
            if classInNum[j] != 3:
                mid = onsetInFrames[j]
                for k in range(mid-neighbor2Include, mid+neighbor2Include+1):
                    curIndex = k
                    if curIndex < 0:
                        curIndex = 0
                    if curIndex >= np.size(features, axis=1):
                        curIndex = np.size(features, axis=1) - 1
                    featureSlice = features[:, curIndex]
                    X.append(featureSlice)
                    y.append(classInNum[j])
                    originalFilePath.append(audioPath)
    print(np.shape(X))
    print(np.shape(y))
    print('saving results to %s' % savePath)
    np.savez(savePath, X, y, originalFilePath)
    return()

'''
onsetInFrames, classInNum = parseAnnotations(annPath)
input:
    annPath: str, path to the annotation files (.txt)
output:
    onsetFrameList: list of onset time in frames (using the hop size and sampling rate defined)
    classNumList: list of drum classes in numbers (see the following notes)
Note:
    0: kick drum
    1: snare drum
    2: hihat
    3: others
'''
def parseAnnotations(annPath):
    onsetFrameList = []
    classNumList = []
    annFile = open(annPath, 'r')
    for line in annFile.readlines():
        onset, drum = line.split()
        onsetFrame = int(float(onset)/(HOPSIZE/FS))
        if drum == 'KD' or drum == 'bd' or  drum == '0':
            classNum = 0
        elif drum == 'SD' or drum == 'sd' or drum == '1':
            classNum = 1
        elif drum == 'HH' or drum == 'chh' or drum == 'ohh' or drum == '2':
            classNum = 2
        else:
            classNum = 3
        onsetFrameList.append(onsetFrame)
        classNumList.append(classNum) 
    return onsetFrameList, classNumList 

'''
savePath = getSavePath(dataList, saveFolder, featureOptions)
input:
    dataList: str, path to the data list
    saveFolder: str, the folder where the extracted features will be stored
    featureOptions: str, what kind of features (e.g., baseline, convAe)
output:
    savePath: str, generated savePath for the feature matrix
'''
def getSavePath(dataList, saveFolder, featureOptions):
    listName = dataList.split('/')[-1].split('.')[0]
    savePath = saveFolder + listName + '_' + featureOptions + '.npz'
    return savePath

def main():
    allDataList = ['./dataLists/enstList.npy', './dataLists/mdbList.npy', './dataLists/rbmaList.npy', './dataLists/m2005List.npy']
    allFeatureOptions = ['baseline', 'convRandom']
    for dataList in allDataList:
        for featureOption in allFeatureOptions:
            savePath = getSavePath(dataList, './featureMat/', featureOption)
            getFeatureMatrix(dataList, savePath, featureOption)
    return()

if __name__ == "__main__":
    print('running main() directly!')
    main()
