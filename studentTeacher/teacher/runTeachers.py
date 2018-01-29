'''
this script applies different systems on the unlabeled data and generate soft targets
CW @ GTCMT 2018
'''
import numpy as np
import time
import sys
from os.path import isdir, isfile
from os import mkdir, makedirs
from pfNmf import pfNmf
from ADTLib import ADT
from scipy import signal


HOPSIZE = 512
WINSIZE = 2048
FS = 44100.0


def runPfNmf(sourceLists, saveFolder, templatePath):
    if not isdir(saveFolder):
        print('designated folder does not exist, creating now...')
        mkdir(saveFolder)
    sourceTrain, sourceVal, sourceTest = unpackSourceLists(sourceLists)
    WD = np.load(templatePath)
    print('Processing %d training files' % len(sourceTrain))
    for sourceFilePath, genre in sourceTrain:
        #==== handle str
        print('working on %s' % sourceFilePath)
        print('template = %s' % templatePath)
        sourceFilepathAdjusted = '../../preprocessData' + sourceFilePath[1:]
        saveFolderAdjusted = saveFolder + 'training/' + genre + '/'
        if not isdir(saveFolderAdjusted):
            makedirs(saveFolderAdjusted)
        saveFilePath = saveFolder + 'training/' + sourceFilePath.split('./stft/')[1]
        
        #==== perform NMF
        if isfile(saveFilePath):
            print('file exists, check nan!')
            HD = np.load(saveFilePath)
            if checkNan(HD):
                print('nan exists, running pfnmf again...')
                X = np.load(sourceFilepathAdjusted)
                X = abs(X)
                tic = time.time()
                WD, HD, WH, HH, err = pfNmf(X, WD, HD=[], WH=[], HH=[], rh=50, sparsity=0.0)
                print((time.time()-tic))
                print('done, saving the results')
                np.save(saveFilePath, HD)
            else:
                print('no nan, moving on!')
        else:
            X = np.load(sourceFilepathAdjusted)
            X = abs(X)
            tic = time.time()
            WD, HD, WH, HH, err = pfNmf(X, WD, HD=[], WH=[], HH=[], rh=50, sparsity=0.0)
            print((time.time()-tic))
            np.save(saveFilePath, HD)
    return ()

def runAdtlib(sourceLists, saveFolder):
    if not isdir(saveFolder):
        print('designated folder does not exist, creating now...')
        mkdir(saveFolder)
    sourceTrain, sourceVal, sourceTest = unpackSourceLists(sourceLists)
    print('Processing %d training files' % len(sourceTrain))
    for sourceFilePath, genre in sourceTrain:
        #==== handle str
        stftSourcePath = '../../preprocessData' + sourceFilePath[1:]
        adjustedSourcePath = '/data/unlabeledDrumDataset/audio/' + sourceFilePath.split('./stft/')[1].split('.npy')[0] + '.mp3'
        saveFolderAdjusted = saveFolder + 'training/' + genre + '/'
        if not isdir(saveFolderAdjusted):
            makedirs(saveFolderAdjusted)
        saveFilePath = saveFolder + 'training/' + sourceFilePath.split('./stft/')[1]

        if isfile(saveFilePath):
            print('file exists, next!')
        else:
            print('working on %s' % adjustedSourcePath)
            X = np.load(stftSourcePath)
            numFreq, numBlock = np.shape(X)
            predictions, allActiv = useAdtlibSingleFile(adjustedSourcePath)
            allActiv = cropAdtlibResults(allActiv, duration=29, startLoc='middle')
            assert(numBlock == np.size(allActiv, 1))
            np.save(saveFilePath, allActiv)
    return ()


'''
input:
    inputTensor: float, ndarray
output:
    Bool: whether inputTensor contains inf or not
'''
def checkInf(inputTensor):
    infCount = 0
    for element in np.nditer(inputTensor):
        if np.isinf(element):
            infCount += 1

    if infCount > 0:
        return True
    else:
        return False

'''
input:
    inputTensor: float, ndarray
output:
    Bool: whether inputTensor contains nan or not
'''
def checkNan(inputTensor):
    nanCount = 0
    for element in np.nditer(inputTensor):
        if np.isnan(element):
            nanCount += 1

    if nanCount > 0:
        return True
    else:
        return False

def useAdtlibSingleFile(audioPath):
    onsets, activations = ADT([audioPath], text='no', tab='no', save_dir='none', output_act='yes')
    activ = activations[0]
    activKd = activ[0][:, 0]
    activSd = activ[1][:, 0]
    activHh = activ[2][:, 0]
    activKd = np.expand_dims(activKd, axis=0)
    activSd = np.expand_dims(activSd, axis=0)
    activHh = np.expand_dims(activHh, axis=0)
    allActiv = np.concatenate((activKd, activSd, activHh), axis=0)

    onsetsSingleTrack = onsets[0]
    onsetsKd = onsetsSingleTrack['Kick']
    onsetsSd = onsetsSingleTrack['Snare']
    onsetsHh = onsetsSingleTrack['Hihat']

    allOnsetLists = [onsetsKd, onsetsSd, onsetsHh]
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
    return predictions, allActiv

def cropAdtlibResults(allActiv, duration, startLoc):
    durationInBlocks = round(float(duration)/(HOPSIZE/FS))
    numDrums, numBlock = np.shape(allActiv)
    if startLoc == 'beginning':
        istart = 0
    elif startLoc == 'middle':
        istart = round(numBlock/2)
    iend = istart + durationInBlocks

    if iend > np.size(allActiv, 1):
        istart = 0
        iend = istart + durationInBlocks
    if iend > np.size(allActiv, 1):
        istart = 0
        iend = istart + durationInBlocks
        gap = iend - np.size(allActiv, 1)
        zeropad = np.zeros((3, gap))
        allActiv = np.concatenate((allActiv, zeropad), axis=1)
    allActivCropped = allActiv[:, int(istart):int(iend)] 
    return allActivCropped

def unpackSourceLists(sourceLists):
    sourceAllLists = np.load(sourceLists)
    sourceTrain = sourceAllLists[0]
    sourceVal = sourceAllLists[1]
    sourceTest = sourceAllLists[2]
    return sourceTrain, sourceVal, sourceTest

def main():
    sourceLists = '../../preprocessData/stft_train_test_splits.npy'
    allTeachers = ['pfnmf_200d', 'pfnmf_smt']#['pfnmf_200d', 'pfnmf_smt', 'adtlib']

    #==== check saving folder
    if not isdir('./softTargets/'):
        print('designated folder does not exist, creating now...')
        mkdir('./softTargets/')
    
    #==== go through all teacher systems
    for teacher in allTeachers:
        print('teacher = %s' % teacher)
        saveFolder = './softTargets/' + teacher + '/'
        if teacher == 'pfnmf_200d':
            templatePath = './drumTemplates/template_200drums_2048_512.npy'
            runPfNmf(sourceLists, saveFolder, templatePath)
        elif teacher == 'pfnmf_smt':
            templatePath = './drumTemplates/template_smt_2048_512.npy'
            runPfNmf(sourceLists, saveFolder, templatePath)
        elif teacher == 'adtlib':
            runAdtlib(sourceLists, saveFolder)
    return ()

if __name__ == "__main__":
    print('running main() directly')
    main()

