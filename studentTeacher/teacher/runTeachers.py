'''
this script applies different systems on the unlabeled data and generate soft targets
CW @ GTCMT 2018
'''
import numpy as np
from os.path import isdir, isfile
from os import mkdir, makedirs
from pfNmf import pfNmf
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


def unpackSourceLists(sourceLists):
    sourceAllLists = np.load(sourceLists)
    sourceTrain = sourceAllLists[0]
    sourceVal = sourceAllLists[1]
    sourceTest = sourceAllLists[2]
    return sourceTrain, sourceVal, sourceTest

def main():
    sourceLists = '../../preprocessData/stft_train_test_splits.npy'
    allTeachers = ['pfnmf_200d', 'pfnmf_smt']

    #==== check saving folder
    if not isdir('./softTargets/'):
        print('designated folder does not exist, creating now...')
        mkdir('./softTargets/')
    
    #==== go through all teacher systems
    for teacher in allTeachers:
        saveFolder = './softTargets/' + teacher + '/'
        if teacher == 'pfnmf_200d':
            templatePath = './drumTemplates/template_200drums_2048_512.npy'
            runPfNmf(sourceLists, saveFolder, templatePath)
        elif teacher == 'pfnmf_smt':
            templatePath = './drumTemplates/template_smt_2048_512.npy'
            runPfNmf(sourceLists, saveFolder, templatePath)
    return ()

if __name__ == "__main__":
    print('running main() directly')
    main()

