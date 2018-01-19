'''
this script applies different systems on the unlabeled data and generate soft targets
CW @ GTCMT 2018
'''
import numpy as np
from os.path import isdir, isfile
from os import mkdir, makedirs
from pfNmf import pfNmf
from ADTLib import ADT
from scipy import signal
import time

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
            print('file exists, next!')
        else:
            X = np.load(sourceFilepathAdjusted)
            X = abs(X)
            tic = time.time()
            WD, HD, WH, HH, err = pfNmf(X, WD, HD=[], WH=[], HH=[], rh=50, sparsity=0.0)
            print((time.time()-tic))
            np.save(saveFilePath, HD)
    
    # print('Processing %d validation files' % len(sourceVal))
    # for sourceFilePath, genre in sourceVal:
    #     print('working on %s' % sourceFilePath)
    #     sourceFilepathAdjusted = '../../preprocessData' + sourceFilePath[1:]
    #     X = np.load(sourceFilepathAdjusted)
    #     X = abs(X)
    #     WD, HD, WH, HH, err = pfNmf(X, WD, HD=[], WH=[], HH=[], rh=50, sparsity=0.0)
    #     saveFilePath = saveFolder + 'validation/' + sourceFilePath.split('./stft/')[1]
    #     np.save(saveFilePath, HD)
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
            print(np.shape(X))
            allActiv = useAdtlibSingleFile(adjustedSourcePath, numBlock)
            np.save(saveFilePath, allActiv)
    return ()

def useAdtlibSingleFile(audioPath, targetNumBlock):
    onsets, activations = ADT([audioPath], text='no', tab='no', save_dir='none', output_act='yes')
    activ = activations[0]
    activKd = activ[0][:, 0]
    activSd = activ[1][:, 0]
    activHh = activ[2][:, 0]
    activKd_resampled = signal.resample(activKd, num=targetNumBlock)
    activSd_resampled = signal.resample(activSd, num=targetNumBlock)
    activHh_resampled = signal.resample(activHh, num=targetNumBlock)
    activKd_resampled = np.expand_dims(activKd_resampled, axis=0)
    activSd_resampled = np.expand_dims(activSd_resampled, axis=0)
    activHh_resampled = np.expand_dims(activHh_resampled, axis=0)
    allActiv = np.concatenate((activKd_resampled, activSd_resampled, activHh_resampled), axis=0)
    return allActiv



def unpackSourceLists(sourceLists):
    sourceAllLists = np.load(sourceLists)
    sourceTrain = sourceAllLists[0]
    sourceVal = sourceAllLists[1]
    sourceTest = sourceAllLists[2]
    return sourceTrain, sourceVal, sourceTest

def main():
    sourceLists = '../../preprocessData/stft_train_test_splits.npy'
    allTeachers = ['adtlib']#['pfnmf_200d', 'pfnmf_smt', 'adtlib']

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

