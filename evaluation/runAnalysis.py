'''
this script is to analyze the evaluation results and try to highlight the files with largest improvements (or decreasements)
CW @ GTCMT 2018
'''
import numpy as np
from scipy.stats import ttest_rel


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
            testValue = testLineArr[7]
            refValue  = refLineArr[7]
            testPreValue = testLineArr[8]
            refPreValue = refLineArr[8]
            testRecValue = testLineArr[9]
            refRecValue = refLineArr[9]
            diffValue = testValue - refValue
        elif instrument == 'bd':
            testValue = testLineArr[4]
            refValue  = refLineArr[4]
            testPreValue = testLineArr[5]
            refPreValue = refLineArr[5]
            testRecValue = testLineArr[6]
            refRecValue = refLineArr[6]
            diffValue = testValue - refValue
        
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
    margin = diffFVec[songIndex]
    tscore, pvalue = ttest_rel(testFVec, refFVec)
    
    print('song index = %d' % songIndex)
    print('file name = %s' % songFilePath)
    print('margin between 2 systems = %f' % margin)
    print('p value of the ttest = %f' % pvalue)

    print('test F = %f, P = %f, R = %f' % (testFVec[songIndex], testPreVec[songIndex], testRecVec[songIndex]))
    print('ref F = %f, P = %f, R = %f' % (refFVec[songIndex], refPreVec[songIndex], refRecVec[songIndex]))
    return ()



def main():
    # testFilePath = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/FC_200_all_genres_100ep/studentTeacher_enst_feat_FC_200_predictionList.txt'
    # refFilePath = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/pfnmf_smt/studentTeacher_enst_feat_pfnmf_smt_predictionList.txt'
    # dataList = '/home/cwu/ADT_with_unlabeledData/featureLearning/featureExtraction/dataLists/enstList.npy'

    # testFilePath = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/FC_200_all_genres_100ep/studentTeacher_m2005_feat_FC_200_predictionList.txt'
    # refFilePath = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/pfnmf_smt/studentTeacher_m2005_feat_pfnmf_smt_predictionList.txt'
    # dataList = '/home/cwu/ADT_with_unlabeledData/featureLearning/featureExtraction/dataLists/m2005List.npy'

    # testFilePath = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/FC_200_all_genres_100ep/studentTeacher_mdb_feat_FC_200_predictionList.txt'
    # refFilePath = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/pfnmf_smt/studentTeacher_mdb_feat_pfnmf_smt_predictionList.txt'
    # dataList = '/home/cwu/ADT_with_unlabeledData/featureLearning/featureExtraction/dataLists/mdbList.npy'

    testFilePath = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/FC_200_all_genres_100ep/studentTeacher_rbma_feat_FC_200_predictionList.txt'
    refFilePath = '/home/cwu/ADT_with_unlabeledData/evaluation/archivedEvalResults/pfnmf_smt/studentTeacher_rbma_feat_pfnmf_smt_predictionList.txt'
    dataList = '/home/cwu/ADT_with_unlabeledData/featureLearning/featureExtraction/dataLists/rbmaList.npy'

    instrument = 'hh' # 'sd', 'bd'
    songID = compareResults(testFilePath, refFilePath, instrument, dataList)
    return ()


if __name__ == "__main__":
    print('running main() directly')
    main()

