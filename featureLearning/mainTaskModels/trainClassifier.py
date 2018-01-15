'''
this script is to train classifier for the main task (automatic drum transcription/classification)
CW @ GTCMT 2017
'''
import numpy as np
import sys
import time
sys.path.insert(0, '../autoencoder')
sys.path.insert(0, '../featureExtraction')
from extractFeatures import checkNan
from FileUtil import scaleMatrix, scaleMatrixWithMinMax, zscoreMatrix, zscoreMatrixWithAvgStd
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
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

'''
input:
    all: before concatenation 2D matrix
    new: incoming 2D matrix
output:
    all: after concatenation 2D matrix, concatenated along axis=0
'''
def concateMatrix(all, new):
    if len(all) == 0:
        all = new
    else:
        all = np.concatenate((all, new), axis=0)
    return all

'''
input:
    y_original: the original multi-classes label array
    targetAnn: target drum
                0: bd
                1: sd
                2: hh
output:
    y_target: binary label array (target annotation = 1, others = 0)
'''
def adjustTarget4Instruments(y_original, targetAnn):
    y_target = np.zeros(np.shape(y_original))
    for i in range(0, len(y_original)):
        if y_original[i] == targetAnn:
            y_target[i] = 1
    return y_target


'''
this function performs grid search on the SVM parameter space. The best classifier is selected based on the 10-fold cross-validation accuracy
input:
    X: training data, numSample x numFeature
    y: label, numSample
output:
    bestClassifier
'''
def gridSearchClassifier(X, y):
    bestClassifier = []

    #==== Linear SVM (commented out for experimenting)
    param_grid = {'C':[0.1, 1.0, 10.0, 100.0, 1000.0], 'dual':[False], 'class_weight':['balanced']} 
    svm = LinearSVC()
    tic = time.time()
    clf = GridSearchCV(svm, param_grid=param_grid, cv=10, refit=True)
    clf.fit(X, y)
    bestClassifier = clf.best_estimator_
    print('time elapsed %f' % (time.time()-tic))

    #==== Random Forest (commented out for experimenting)
    # rf = RandomForestClassifier()
    # param_grid = {'n_estimators':[10, 50], 'max_depth':[2, 4], 'class_weight':['balanced_subsample']}
    # tic = time.time()
    # clf = GridSearchCV(rf, param_grid=param_grid, cv=10, refit=True)
    # clf.fit(X, y)
    # bestClassifier = clf
    # print('time elapsed %f' % (time.time()-tic))

    cvBestScore = clf.best_score_
    print('best cv score (accuracy) = %f' % cvBestScore)
    return bestClassifier

'''
input:
    X: ndarray, training data numSample x numFeature
    y: ndarray, training label 
output:
    classifiers: array with 3 classifiers in the following order [bd, sd, hh] 
    normParams: array with parameters for normalization [maxVec, minVec]
'''
def getAllClassifiers(X, y):
    # shuffle data
    # train classifiers for different drums
    # save all models 
    classifiers = []
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.15, random_state=33)
    # sub-sampling the training set (test_size determines the sub-sampling size)
    dump, XTrain, dump, yTrain = train_test_split(XTrain, yTrain, test_size=0.10, random_state=33)
    print('there are %d samples in the training set' % len(yTrain))
    XTrainScaled, maxVec, minVec = scaleMatrix(XTrain)
    yBdTrain = [ele[0] for ele in yTrain]
    ySdTrain = [ele[1] for ele in yTrain]
    yHhTrain = [ele[2] for ele in yTrain]

    # print(maxVec)
    # print(minVec)
    
    XTestScaled = scaleMatrixWithMinMax(XTest, maxVec, minVec)
    yBdTest = [ele[0] for ele in yTest]
    ySdTest = [ele[0] for ele in yTest]
    yHhTest = [ele[0] for ele in yTest]

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
        if drum[0] == 1:
            bdCount += 1
        if drum[1] == 1:
            sdCount += 1
        if drum[2] == 1:
            hhCount += 1
        if np.sum(drum) == 0:
            otCount += 1
    print('bd = %d sd = %d hh = %d other = %d' % (bdCount, sdCount, hhCount, otCount))
    return()

def main():
    saveFolder = './trainedClassifier/'
    allFeatureOptions = ['convDae']#['baseline', 'convRandom']#, 'convAe', 'convDae']
    allHeldOutOptions = ['enst', 'mdb', 'rbma', 'm2005']
    
    for featureOption in allFeatureOptions:
        for heldOutOption in allHeldOutOptions:
            X_final, y_final, filelist_final = prepareTrainingData(heldOutOption, featureOption)
            print('====================================')
            print('current feature = %s'% featureOption)
            print('current held out = %s'% heldOutOption)
            #summarizeClassDistribution(y_final)
            classifiers, normParams = getAllClassifiers(X_final, y_final)
            classifierPath = getClassifierPath(heldOutOption, featureOption, saveFolder) 
            np.savez(classifierPath, classifiers, normParams)  
      
            # #==== quick sanity check
            # print('====================================')
            # print('test on the held out dataset %s'% heldOutOption)
            # heldOutDataPath = '../featureExtraction/featureMat/' + heldOutOption + 'List_' + featureOption + '.npz'
            # tmp = np.load(heldOutDataPath)
            # X = tmp['arr_0']
            # y = tmp['arr_1']
            # fileList = tmp['arr_2'] 
            # #summarizeClassDistribution(y)

            # tmp = np.load(classifierPath)
            # classifiers = tmp['arr_0']
            # normParams = tmp['arr_1']            
            # clfBd = classifiers[0]
            # clfSd = classifiers[1]
            # clfHh = classifiers[2]

            # maxVec = normParams[0]
            # minVec = normParams[1]
            # X = scaleMatrixWithMinMax(X, maxVec, minVec)

            # yBdTest = [ele[0] for ele in y]
            # ySdTest = [ele[1] for ele in y]
            # yHhTest = [ele[2] for ele in y]

            # bdScore = clfBd.score(X, yBdTest)
            # sdScore = clfSd.score(X, ySdTest)
            # hhScore = clfHh.score(X, yHhTest)

            # print(bdScore)
            # print(sdScore)
            # print(hhScore)
    return()

if __name__ == "__main__":
    print('running main() directly')
    main()