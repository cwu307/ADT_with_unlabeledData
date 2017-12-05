'''
this script is to train classifier for the main task (automatic drum transcription/classification)
CW @ GTCMT 2017
'''
import numpy as np
import sys
import time
sys.path.insert(0, '../autoencoder')

from FileUtil import scaleMatrix, scaleMatrixWithMinMax
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

'''
input:
    heldOutOption: str, viable options are 'enst', 'mdb', 'rbma', 'm2005'
    featureOption: str, viable options are 'baseline', 'convAe', 'convDae', 'convRandom'
    Note: the .npy file in /featureMat/ directory has the following format [X, y, originalFilePath]
output:
    X_final: ndarray, numSample x numFeature, the final concatenated feature matrix
    y_final: ndarray, numSample, the final concatenated target vector
    fileList_final: ndarray, numSample, the corresponding file paths
'''
def prepareTrainingData(heldOutOption, featureOption):
    allData = ['enst', 'mdb', 'rbma', 'm2005']
    allData.remove(heldOutOption)
    X_final = []
    y_final = []
    fileList_final = []
    for dataset in allData:
        dataPath = '../featureExtraction/featureMat/' + dataset + 'List_' + featureOption + '.npz'
        tmp = np.load(dataPath)
        X = tmp['arr_0']
        y = tmp['arr_1']
        fileList = tmp['arr_2'] 
        X_final = concateMatrix(X_final, X)
        y_final = concateMatrix(y_final, y)
        fileList_final = concateMatrix(fileList_final, fileList)
    return X_final, y_final, fileList_final 


def concateMatrix(all, new):
    if len(all) == 0:
        all = new
    else:
        all = np.concatenate((all, new), axis=0)
    return all


def adjustTarget4Instruments(y_original, targetAnn):
    y_target = np.zeros(np.shape(y_original))
    for i in range(0, len(y_original)):
        if y_original[i] == targetAnn:
            y_target[i] = 1
    return y_target

def gridSearchClassifier(X, y):
    bestClassifier = []
    #param_grid = {'C':[0.1, 1.0, 10.0, 100.0, 1000.0], 'kernel':['rbf'], 'gamma':[1.0/np.power(2, 3), 1.0/np.power(2, 5), 1.0/np.power(2, 7), 1.0/np.power(2, 9), 'auto']}
    param_grid = {'C':[0.1, 1.0, 10.0, 100.0, 1000.0], 'kernel':['linear'], 'max_iter':[-1]}
    svm = SVC()
    tic = time.time()
    clf = GridSearchCV(svm, param_grid=param_grid, cv=10, refit=True)
    clf.fit(X, y)
    print('time elapsed %f' % (time.time()-tic))
    bestClassifier = clf.best_estimator_
    cvBestScore = clf.best_score_
    print('best cv score (accuracy) = %f' % cvBestScore)
    return bestClassifier


def getAllClassifiers(X, y):
    # shuffle data
    # train classifiers for different drums
    # save all models 
    classifiers = []
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.20, random_state=33)
    # sub-sampling the training set
    # dump, XTrain, dump, yTrain = train_test_split(XTrain, yTrain, test_size=0.05, random_state=33)
    print(len(yTrain))
    XTrainScaled, maxVec, minVec = scaleMatrix(XTrain)
    yBdTrain = adjustTarget4Instruments(yTrain, 0)
    ySdTrain = adjustTarget4Instruments(yTrain, 1)
    yHhTrain = adjustTarget4Instruments(yTrain, 2)
    
    XTestScaled = scaleMatrixWithMinMax(XTest, maxVec, minVec)
    yBdTest = adjustTarget4Instruments(yTest, 0)
    ySdTest = adjustTarget4Instruments(yTest, 1)
    yHhTest = adjustTarget4Instruments(yTest, 2)

    print('==== grid search on BD...')
    clfBestBd = gridSearchClassifier(XTrainScaled, yBdTrain)
    print('test accuracy = %f' % clfBestBd.score(XTestScaled, yBdTest))
    print('==== grid search on SD...')
    clfBestSd = gridSearchClassifier(XTrainScaled, ySdTrain)
    print('test accuracy = %f' % clfBestSd.score(XTestScaled, ySdTest))
    print('==== grid search on HH...')
    clfBestHh = gridSearchClassifier(XTrainScaled, yHhTrain)
    print('test accuracy = %f' % clfBestHh.score(XTestScaled, yHhTest))

    #return the best models
    classifiers = [clfBestBd, clfBestSd, clfBestHh]
    normParams = [maxVec, minVec]
    return classifiers, normParams

def getClassifierPath(heldOutOption, featureOption, saveFolder):
    classifierPath = saveFolder + 'heldout_' + heldOutOption + '_feat_' + featureOption + '.npz'
    return classifierPath

def summarizeClassDistribution(y):
    print('there are %d samples in current set' % len(y))
    hhCount = 0
    bdCount = 0
    sdCount = 0
    otCount = 0
    for drum in y:
        if drum == 0:
            bdCount += 1
        elif drum == 1:
            sdCount += 1
        elif drum == 2:
            hhCount += 1
        else:
            otCount += 1
    print('bd = %d sd = %d hh = %d other = %d' % (bdCount, sdCount, hhCount, otCount))
    return()

def main():
    saveFolder = './trainedClassifier/'
    allFeatureOptions = ['baseline']#, 'convRandom']#, 'convAe', 'convDae']
    allHeldOutOptions = ['enst'] #, 'mdb', 'rbma', 'm2005']
    
    for featureOption in allFeatureOptions:
        for heldOutOption in allHeldOutOptions:
            X_final, y_final, filelist_final = prepareTrainingData(heldOutOption, featureOption)
            summarizeClassDistribution(y_final)
            classifiers, normParams = getAllClassifiers(X_final, y_final)
            classifierPath = getClassifierPath(heldOutOption, featureOption, saveFolder) 
            np.savez(classifierPath, classifiers, normParams)   
    return()

if __name__ == "__main__":
    print('running main() directly')
    main()