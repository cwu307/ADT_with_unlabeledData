'''
this script applies different systems on the unlabeled data and generate soft targets
CW @ GTCMT 2018
'''
import numpy as np
from os.path import isdir
from os import mkdir
from pfNmf import pfNmf

def runPfNmf(sourceLists, saveFolder, templatePath):
    if not isdir(saveFolder):
        print('designated folder does not exist, creating now...')
        mkdir(saveFolder)
    sourceTrain, sourceVal, sourceTest = unpackSourceLists(sourceLists)
    WD = np.load(templatePath)
    for sourceFilePath, genre in sourceTrain:
        sourceFilepathAdjusted = '../../preprocessData' + sourceFilePath[1:]
        X = np.load(sourceFilepathAdjusted)
        X = abs(X)
        WD, HD, WH, HH, err = pfNmf(X, WD, HD=[], WH=[], HH=[], rh=50, sparsity=0.0)
        saveFilePath = saveFolder + 'training/' + sourceFilePath.split('./stft/')[1]
        np.save(saveFilePath, HD)
    
    for sourceFilePath, genre in sourceVal:
        sourceFilepathAdjusted = '../../preprocessData' + sourceFilePath[1:]
        X = np.load(sourceFilepathAdjusted)
        X = abs(X)
        WD, HD, WH, HH, err = pfNmf(X, WD, HD=[], WH=[], HH=[], rh=50, sparsity=0.0)
        saveFilePath = saveFolder + 'validation/' + sourceFilePath.split('./stft/')[1]
        np.save(saveFilePath, HD)
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

