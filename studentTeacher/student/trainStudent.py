'''
this script is to train the student models using the unlabeled data 
and soft-targets generated from the teachers
CW @ GTCMT 2018
'''
import numpy as np
import sys
sys.path.insert(0, '../teacher/')
sys.path.insert(0, '../../featureLearning/autoencoder/')
sys.path.insert(0, '../../featureLearning/featureExtraction/')
from extractFeatures import convert2dB, scaleTensorTrackwise, checkNan, prepareConvnetInput
from librosa.feature import melspectrogram
from FileUtil import scaleMatrixWithMinMax, scaleMatrix
from runTeachers import unpackSourceLists
from studentDnnModels import createFcModel, createCnnModel
from os import mkdir, makedirs
from os.path import isdir, isfile
from keras.optimizers import Adam, rmsprop, adagrad
from keras.models import load_model
from keras.callbacks import TensorBoard, EarlyStopping
from tensorboard_logger import configure, log_value
CONTFLAG = False

'''
input:
    sourceTrain: list, a list of tuples (sourceFilePath, genre)
    studentModelPath: str, path to a student model
    studentModelType: str, type of DNNs as student model {'FC', 'CNN', 'RNN'}
    teacher: str, name of the teacher system 
    normParamPath: str, path to normalization parameters for activation functions (soft-targets)
output:
    N/A, but the trained student model will be store to studentModelSavePath as defined by the user
'''
def trainStudent(sourceTrain, studentFullPath, studentType, teacher, normParamPath):
    numEpoch = 30
    batchSize = 32
    
    #==== load DNN model
    print('%s model is selected' % studentType)
    studentModel = load_model(studentFullPath)

    #==== start training
    tmp = np.load(normParamPath)
    maxVec = tmp['arr_0']
    minVec = tmp['arr_1']

    c = 0
    for e in range(0, numEpoch):
        print('==== Epoch = %d ===='% e)
        allTrainLoss = []
        for sourceFilePath, genre in sourceTrain:
            #==== handle str
            print('working on %s' % sourceFilePath)
            print('count = %d' % c)
            sourceFilepathAdjusted = '../../preprocessData' + sourceFilePath[1:]
            targetFilepath = '../teacher/softTargets/' + teacher + '/training/'+ sourceFilePath.split('./stft/')[1]

            X = np.load(sourceFilepathAdjusted)
            X = abs(X)
            X = np.transpose(X)
            y = np.load(targetFilepath)
            y = np.transpose(y)

            #==== normalize activation functions to {0, 1} for sigmoid
            yScaled = scaleMatrixWithMinMax(y, maxVec, minVec)

            #==== convert X and yScaled if needed
            if studentType == 'CNN':
                batchSize = 19
                X, yScaled = prepareCnnInOut(X, yScaled)

            #==== training 
            studentModel.fit(X, yScaled, batch_size=batchSize, epochs=1, verbose=0, shuffle=True)
            results = studentModel.evaluate(X, yScaled, batch_size=32, verbose=1)
            trainLoss = results[0]
            print('training loss = %f' % trainLoss)
            allTrainLoss.append(trainLoss)
            c += 1

        print('logging training loss...')
        log_value('training_loss', np.mean(allTrainLoss), e)
        print('saving model at epoch = %d'% e)
        studentModel.save(studentFullPath)
    return()

'''
quick implementation to reproduce my ISMIR results...
first concatenate all the magnitude spectrogram / soft targets within the same genre
then perform the training using a larger batch size 
'''
def trainStudentConcate(sourceTrain, studentFullPath, studentType, teacher, normParamPath):
    numEpoch = 300
    batchSize = 640
    
    #==== load DNN model
    print('%s model is selected' % studentType)
    studentModel = load_model(studentFullPath)
    tbCallBack = TensorBoard(log_dir='./tblogs/FC/', histogram_freq=0, write_graph=True, write_images=True)
    #earlyStopCallBack = EarlyStopping(monitor='loss', min_delta=1e-6, patience=10)

    #==== start training
    tmp = np.load(normParamPath)
    maxVec = tmp['arr_0']
    minVec = tmp['arr_1']

    XAll = np.zeros((0, 1025))
    yAll = np.zeros((0, 3))
    counter = 0
    for sourceFilePath, genre in sourceTrain:
        #==== handle str
        print('working on %s' % sourceFilePath)
        sourceFilepathAdjusted = '../../preprocessData' + sourceFilePath[1:]
        targetFilepath = '../teacher/softTargets/' + teacher + '/training/'+ sourceFilePath.split('./stft/')[1]

        X = np.load(sourceFilepathAdjusted)
        X = abs(X)
        X = np.transpose(X)
        y = np.load(targetFilepath)
        y = np.transpose(y)
        #==== normalize activation functions to {0, 1} for sigmoid
        y = scaleMatrixWithMinMax(y, maxVec, minVec)
        if np.any(y > 1.0):
            print('outlier found, skip this track')
        else:
            if counter < 200:
                XAll = np.concatenate((XAll, X), axis=0)
                yAll = np.concatenate((yAll, y), axis=0)
                counter += 1
            else:
                #==== training 
                studentModel.fit(XAll, yAll, batch_size=batchSize, epochs=numEpoch, callbacks=[tbCallBack])
                studentModel.save(studentFullPath)
                print('counter reset')
                XAll = np.zeros((0, 1025))
                yAll = np.zeros((0, 3))
                counter = 0

    print(np.shape(yAll))
    #==== training 
    studentModel.fit(XAll, yAll, batch_size=batchSize, epochs=numEpoch, callbacks=[tbCallBack])
    studentModel.save(studentFullPath)
    return()

'''
this function estimates the parameters for normalizing the soft-targets from the teacher systems
input:
    sourceLists: str, path to the data splits which contain training, validation, and test list
    teacher: str, e.g. 'pfnmf_200d' or 'pfnmf_smt'
    normParamPath: str, path for saving the normalization parameters
output:
    N/A, but the parameters will be saved in the current folder as 'normParam_pfnmf_200d.npz' for example
'''
def estimateActivNormParam(sourceTrain, teacher, normParamPath):
    yAllMax = np.ndarray((0, 3))
    yAllMin = np.ndarray((0, 3))
    c = 0
    nanFiles = []
    for sourceFilePath, genre in sourceTrain:
        c += 1
        print(c)
        targetFilePath = '../teacher/softTargets/' + teacher + '/training/'+ sourceFilePath.split('./stft/')[1]
        y = np.load(targetFilePath)
        y = np.transpose(y)
        if checkNan(y):
            nanFiles.append(sourceFilePath)
        yMax = np.expand_dims(np.max(y, axis=0), axis=0)
        yMin = np.expand_dims(np.min(y, axis=0), axis=0)
        yAllMax = np.concatenate((yAllMax, yMax), axis=0)
        yAllMin = np.concatenate((yAllMin, yMin), axis=0)

    #==== outlier removal 
    yAllMaxCut = removeOutlierTracks(yAllMax, True)
    yAllMinCut = removeOutlierTracks(yAllMin, False)
    # yAllMaxCut = yAllMax
    # yAllMinCut = yAllMin

    maxVec = np.max(yAllMaxCut, axis=0)
    minVec = np.min(yAllMinCut, axis=0)
    print('%d nan files' % len(nanFiles))
    print(maxVec)
    print(minVec)
    np.savez(normParamPath, maxVec, minVec)
    return ()

def removeOutlierTracks(yAll, reverseFlag):
    #==== remove 1% of total number of songs
    numOutliers = int(np.size(yAll, axis=0) * 0.01)
    for i in range(0, np.size(yAll, 1)):
        yAll[:, i] = sorted(yAll[:, i], reverse=reverseFlag)
    yAllCut = yAll[numOutliers:, :]
    return yAllCut


'''
this function initialize an untrained student DNN model and returns its path
input:
    studentFullPath: str, e.g., './savedStudentModels/FC/studentModel.h5'
    studentType: str, {'FC', 'CNN', 'RNN'}
output:
    None
'''
def initStudentModel(studentFullPath, studentType):
    selectedOptimizer = Adam(lr=0.001)
    selectedLoss = 'mse'
    studentFolder = studentFullPath.split('studentModel.h5')[0]
    if not isdir(studentFolder):
        makedirs(studentFolder)
    if studentType == 'FC' or studentType == 'FCRandom':
        inputDim = 1025
        studentModel = createFcModel(inputDim, selectedOptimizer, selectedLoss)
    elif studentType == 'CNN' or studentType == 'CNNRandom':
        inputDim = 128
        inputDim2 = 128
        studentModel = createCnnModel(inputDim, inputDim2, selectedOptimizer, selectedLoss)

    print('initial untrained model saved at %s', studentFullPath)
    studentModel.save(studentFullPath)
    return ()


'''
this function is to convert the training data into CNN compatible format
input:
    X: ndarray, numBlock x numFreq STFT matrix
    y: ndarray, numBlock x 3 activation matrix
output:
    XCnn: tensor, numSample x 1 x 128 x 128 
    yCnn: tensor, numSample x 3 x 1 x 128
'''
def prepareCnnInOut(X, y):
    X = np.transpose(X)
    y = np.transpose(y)
    XCnn = prepareConvnetInput(X)
    numSample, numCh, numMel, numMel = np.shape(XCnn)
    
    yCnn = np.zeros((0, 3, numMel))
    for i in range(0, numSample):
        istart = i
        iend = istart + numMel
        yCur = y[:, istart:iend]
        yCur = np.expand_dims(yCur, axis=0)
        yCnn = np.concatenate((yCnn, yCur), axis=0)
    yCnn = np.expand_dims(yCnn, axis=2)
    return XCnn, yCnn


'''
this function adjusts the original source file (potentially shorten it)
it regulates the number of songs per genre (max = 1330) without changing the order
input:
    sourceLists: str, a path to a lists of train-test splits
    teacher: str, name of the teacher system 
    selectedGenres: list, available options are ['alternative-songs', 'dance-club-play-songs', 'hot-mainstream-rock-tracks'
                                                'latin-songs', 'pop-songs', 'r-b-hip-hop-songs']
    numSongsPerGenre: int, number of songs per genre
output:
    newSourceTrain: list, a list of tuple of (sourceFilePath, genre) after adjustment
    normParamPath: str, the path to the normalization parameters for activation functionsd
'''
def adjustSourceTrainList(sourceLists, teacher, selectedGenres, numSongsPerGenre):
    sourceTrain, sourceVal, sourceTest = unpackSourceLists(sourceLists)
    numGenres = len(selectedGenres)
    normParamPath = './normParams/normParam_' + teacher + '_genre_' + str(selectedGenres[0]) + '_numSongs' + str(numSongsPerGenre) +'.npz'
    newSourceTrain = []
    fileCountPerGenre = {}
    for genre in selectedGenres:
        fileCountPerGenre[genre] = 0
    for sourceFilePath, genre in sourceTrain:
        if genre in selectedGenres:
            if fileCountPerGenre[genre] < numSongsPerGenre:
                fileCountPerGenre[genre] += 1
                newSourceTrain.append((sourceFilePath, genre))
    return newSourceTrain, normParamPath


def main():
    allTeachers = ['pfnmf_200d', 'pfnmf_smt']
    studentParentFolder = './savedStudentModels/'
    studentType = 'FC' #'FCRandom', 'CNN', 'CNNRandom'
    selectedGenres = ['alternative-songs', 'dance-club-play-songs', 'hot-mainstream-rock-tracks', 'latin-songs', 'pop-songs', 'r-b-hip-hop-songs']
    numSongsPerGenre = 600 #1330
    studentFullPath = studentParentFolder + studentType + '_' + str(numSongsPerGenre) + '/' + 'studentModel.h5'

    #==== init model
    if CONTFLAG:
        print('- continue training an existing model at %s' % studentFullPath)
    else:
        print('- start a new model for training at %s' % studentFullPath)
        initStudentModel(studentFullPath, studentType)

    #==== configure tensorboard logger
    tbPath = './tblogs/' + studentType
    if not isdir(tbPath):
        mkdir(tbPath)
    configure(tbPath)

    #==== loop through all teachers
    for teacher in allTeachers:
        print('==== teacher = %s ====' % teacher)
        for genre in selectedGenres:
            print('==== genre = %s ====' % genre)
            isolatedGenre = [genre]
            sourceLists = '../../preprocessData/stft_train_test_splits.npy' 
            sourceTrain, normParamPath = adjustSourceTrainList(sourceLists, teacher, isolatedGenre, numSongsPerGenre)
            estimateActivNormParam(sourceTrain, teacher, normParamPath)

            print('==== start training student %s model ====' % studentType)
            #trainStudent(sourceTrain, studentFullPath, studentType, teacher, normParamPath)
            trainStudentConcate(sourceTrain, studentFullPath, studentType, teacher, normParamPath)
    return()


if __name__ == "__main__":
    print('running main() directly...')
    main()
    