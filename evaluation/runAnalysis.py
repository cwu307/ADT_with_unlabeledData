'''
this script is to analyze the evaluation results and try to highlight the files with largest improvements (or decreasements)
CW @ GTCMT 2018
'''
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
sys.path.insert(0, '../studentTeacher/')
sys.path.insert(0, '../studentTeacher/teacher/')
from runPrediction import getActivationFromFile

def compareResults(testFilePath, refFilePath, instrument, dataList):
    testFile = open(testFilePath, 'r')
    refFile  = open(refFilePath, 'r')
    testFileData = testFile.readlines()
    refFileData  = refFile.readlines()
    
    diffFVec = []
    testFVec = []
    refFVec  = []
    testPreVec = []
    refPreVec  = []
    testRecVec = []
    refRecVec  = []
    for i in range(1, len(testFileData)-1):
        testLine = testFileData[i]
        testLineArr = [float(ele) for ele in testLine.split('    ')]
        refLine  = refFileData[i]
        refLineArr = [float(ele) for ele in refLine.split('    ')]

        if instrument == 'hh':
            testFValue = testLineArr[10]
            refFValue  = refLineArr[10]
            testPreValue = testLineArr[11]
            refPreValue = refLineArr[11]
            testRecValue = testLineArr[12]
            refRecValue = refLineArr[12]
            diffFValue = testFValue - refFValue
        elif instrument == 'sd':
            testFValue = testLineArr[7]
            refFValue  = refLineArr[7]
            testPreValue = testLineArr[8]
            refPreValue = refLineArr[8]
            testRecValue = testLineArr[9]
            refRecValue = refLineArr[9]
            diffFValue = testFValue - refFValue
        elif instrument == 'bd':
            testFValue = testLineArr[4]
            refFValue  = refLineArr[4]
            testPreValue = testLineArr[5]
            refPreValue = refLineArr[5]
            testRecValue = testLineArr[6]
            refRecValue = refLineArr[6]
            diffFValue = testFValue - refFValue
        
        testFVec.append(testFValue)
        refFVec.append(refFValue)
        testPreVec.append(testPreValue)
        refPreVec.append(refPreValue)
        testRecVec.append(testRecValue)
        refRecVec.append(refRecValue)
        diffFVec.append(diffFValue)

    tmp = np.load(dataList)
    songIndex = np.argmax(diffFVec)
    songFilePath = tmp[songIndex][0]
    annFilePath  = tmp[songIndex][1]
    margin = diffFVec[songIndex]
    tscore, pvalue = ttest_rel(testFVec, refFVec)
    
    print('song index = %d' % songIndex)
    print('file name = %s' % songFilePath)
    print('margin between 2 systems = %f' % margin)
    print('p value of the ttest = %f' % pvalue)

    print('test F = %f, P = %f, R = %f' % (testFVec[songIndex], testPreVec[songIndex], testRecVec[songIndex]))
    print('ref F = %f, P = %f, R = %f' % (refFVec[songIndex], refPreVec[songIndex], refRecVec[songIndex]))
    return songFilePath, annFilePath, diffFVec, testFVec, refFVec



def visualizeComparedSystems(testSystem, refSystem, songFilePath, instrument):
    print('Getting activation function from %s' % testSystem)
    testActiv = getActivationFromFile(songFilePath, testSystem)
    print('Getting activation function from %s' % refSystem)
    refActiv  = getActivationFromFile(songFilePath, refSystem)
    print(np.shape(testActiv))
    print(np.shape(refActiv))

    if instrument == 'bd':
        instIdx = 0
    elif instrument == 'sd':
        instIdx = 1
    elif instrument == 'hh':
        instIdx = 2
    tmp = refActiv[:, instIdx]
    tmp = np.divide(tmp, np.max(abs(tmp)))

    # fig = plt.figure()
    # plt.subplot(211)
    # plt.plot(tmp[0:1000])
    # plt.ylim((0, 0.6))
    # plt.title('HH Activation Function of (Top) Teacher (Bottom) Student')
    # plt.ylabel('Normalized Activity')
    # plt.subplot(212)
    # plt.plot(testActiv[0:1000, instIdx])
    # plt.ylim((0, 0.6))
    # plt.ylabel('Normalized Activity')
    # plt.xlabel('Block Index')
    # plt.savefig('./test.pdf', format='pdf')

    fig = plt.figure()
    plt.plot(tmp[0:1000], 'b')
    plt.plot(testActiv[0:1000, instIdx], 'r')
    plt.ylim((0, 0.6))
    plt.title('HH Activation Function of (blue) Teacher (red) Student')
    plt.ylabel('Normalized Activity')
    plt.xlabel('Block Index')
    plt.savefig('./test.pdf', format='pdf', bbox_inches='tight', dpi=100)
    return ()

def analyzeDiffVec(allDiffFVec):
    positiveVec = []
    negativeVec = []
    for ele in allDiffFVec:
        if ele > 0:
            positiveVec.append(ele)
        elif ele < 0:
            negativeVec.append(ele)
    
    avgPositiveValue = np.mean(positiveVec)
    avgNegativeValue = np.mean(negativeVec)
    return len(positiveVec), len(negativeVec), avgPositiveValue, avgNegativeValue

def main():
    testSystem = 'FC_200'
    refSystem  = 'pfnmf_smt'
    paradigm = 'studentTeacher'
    
    # testSystem = 'convAe'
    # refSystem  = 'baseline'
    # paradigm   = 'featureLearning'
    
    allDatasets = ['enst']#['enst', 'm2005', 'mdb', 'rbma']
    instrument = 'hh' # 'sd', 'bd'
    allDiffFVec = np.zeros((0, 1))
    allTestFVec = np.zeros((0, 1))
    allRefFVec  = np.zeros((0, 1))

    for dataset in allDatasets:
        dataList = '/home/cwu/ADT_with_unlabeledData/featureLearning/featureExtraction/dataLists/' + dataset + 'List.npy'      
        if paradigm is 'studentTeacher':
            testFilePath = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/' + testSystem + '_all_genres_100ep/' + paradigm + '_' + dataset + '_feat_' + testSystem + '_predictionList.txt'
            refFilePath  = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/' + refSystem + '/' + paradigm + '_' + dataset + '_feat_' + refSystem + '_predictionList.txt'     
        elif paradigm is 'featureLearning':
            testFilePath = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/' + paradigm + '_' + testSystem + '/' + paradigm + '_' + dataset + '_feat_' + testSystem + '_predictionList.txt'
            refFilePath  = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/' + paradigm + '_' + refSystem + '/' + paradigm + '_' + dataset + '_feat_' + refSystem + '_predictionList.txt'        
        
        songFilePath, annFilePath, diffFVec, testFVec, refFVec = compareResults(testFilePath, refFilePath, instrument, dataList)
        diffFVec = np.expand_dims(diffFVec, axis=1)
        testFVec = np.expand_dims(testFVec, axis=1)
        refFVec = np.expand_dims(refFVec, axis=1)
        allDiffFVec = np.concatenate((allDiffFVec, diffFVec), axis=0)
        allTestFVec = np.concatenate((allTestFVec, testFVec), axis=0)
        allRefFVec = np.concatenate((allRefFVec, refFVec), axis=0)
        visualizeComparedSystems(testSystem, refSystem, songFilePath, instrument)

    tscore, pvalue = ttest_rel(allTestFVec, allRefFVec)
    positiveCount, negativeCount, avgPositiveValue, avgNegativeValue = analyzeDiffVec(allDiffFVec)
    print(positiveCount)
    print(negativeCount)
    print(avgPositiveValue)
    print(avgNegativeValue)
    print(pvalue)
    return ()

def main_crossParadigmTest():
    testSystem = 'FC_200'
    refSystem  = 'convAe'    
    allDatasets = ['enst', 'm2005', 'mdb', 'rbma']
    instrument = 'bd'#'bd', 'hh'
    allDiffFVec = np.zeros((0, 1))
    allTestFVec = np.zeros((0, 1))
    allRefFVec  = np.zeros((0, 1))

    for dataset in allDatasets:
        dataList = '/home/cwu/ADT_with_unlabeledData/featureLearning/featureExtraction/dataLists/' + dataset + 'List.npy'      
        testFilePath = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/' + testSystem + '_all_genres_100ep/' + 'studentTeacher' + '_' + dataset + '_feat_' + testSystem + '_predictionList.txt'
        refFilePath  = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/featureLearning_' + refSystem + '/' + 'featureLearning' + '_' + dataset + '_feat_' + refSystem + '_predictionList.txt'     
   
        songFilePath, annFilePath, diffFVec, testFVec, refFVec = compareResults(testFilePath, refFilePath, instrument, dataList)
        diffFVec = np.expand_dims(diffFVec, axis=1)
        testFVec = np.expand_dims(testFVec, axis=1)
        refFVec = np.expand_dims(refFVec, axis=1)
        allDiffFVec = np.concatenate((allDiffFVec, diffFVec), axis=0)
        allTestFVec = np.concatenate((allTestFVec, testFVec), axis=0)
        allRefFVec = np.concatenate((allRefFVec, refFVec), axis=0)
        #visualizeComparedSystems(testSystem, refSystem, songFilePath, instrument)

    tscore, pvalue = ttest_rel(allTestFVec, allRefFVec)
    positiveCount, negativeCount, avgPositiveValue, avgNegativeValue = analyzeDiffVec(allDiffFVec)
    print(positiveCount)
    print(negativeCount)
    print(avgPositiveValue)
    print(avgNegativeValue)
    print(pvalue)
    return ()


if __name__ == "__main__":
    print('running main() directly')
    main()

