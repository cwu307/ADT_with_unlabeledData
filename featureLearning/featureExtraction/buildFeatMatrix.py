'''
this script is used to build feature matrices
note: this is used specifically for experimenting the "segment-classify paradigm"
CW @ GTCMT 2017
'''
import numpy as np
import os
from extractFeatures import extractRandomConvFeatures, extractBaselineFeatures, extractConvFeatures, checkNan
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
            modelSavePath = '../autoencoder/savedRandomAeModels/'
            features = extractRandomConvFeatures(audioPath, modelSavePath) #64 x M            
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

        numFeat, numBlock = np.shape(features)

        onsetInFrames, classInNum = parseAnnotations(annPath)
        frontFrame = 0
        rearFrame = 2

        #==== take everything for training
        # X: N by numFeat
        # y: N by 3; list of tuples, each tuple is the binary representation of each drum
        for j in range(0, numBlock):
            splicedFeature = featureSplicing(features, j, frontFrame, rearFrame)
            X.append(splicedFeature)
            yBinary = np.zeros((3,))
            for k in range(0, len(classInNum)):
                if j == onsetInFrames[k]:
                    if classInNum[k] != 3:
                        yBinary[classInNum[k]] = 1 #note: 0=bd, 1=sd, 2=hh
            y.append(yBinary)

    print(np.shape(X))
    print(np.shape(y))
    print('saving results to %s' % savePath)
    np.savez(savePath, X, y, originalFilePath)
    return()


'''
splicedFeature = featureSplicing(features, midIndex, frontFrame, rearFrame)
input:
    features: ndarray feature matrix, numFeature by numBlock
    midIndex: int, the index of current frame (middle)
    frontFrame: int, number of frame to look ahead
    rearFrame: int, number of frame to look after
output:
    splicedFeature: vector, (M,)
    note: the spliced feature would be the concatenation of the previous 
          and latter frames with the middle frame
'''
def featureSplicing(features, midIndex, frontFrame, rearFrame):
    numFeature, numBlock = np.shape(features)
    splicedFeature = []
    for i in range(midIndex - frontFrame, midIndex + rearFrame + 1):
        if i < 0:
            curFrame = features[:, 0]
        elif i >= numBlock:
            curFrame = features[:, numBlock-1]
        else:
            curFrame = features[:, i]
        
        if len(splicedFeature) == 0:
            splicedFeature = curFrame
        else:
            splicedFeature = np.concatenate((splicedFeature, curFrame), axis=0)
    return splicedFeature


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
    allFeatureOptions = ['convDae']#['baseline', 'convRandom']
    for dataList in allDataList:
        for featureOption in allFeatureOptions:
            savePath = getSavePath(dataList, './featureMat/', featureOption)
            getFeatureMatrix(dataList, savePath, featureOption)
    return()

if __name__ == "__main__":
    print('running main() directly!')
    main()
