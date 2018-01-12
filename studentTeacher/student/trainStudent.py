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
from extractFeatures import convert2dB, scaleTensorTrackwise, checkNan
from librosa.feature import melspectrogram
from FileUtil import scaleMatrixWithMinMax
from runTeachers import unpackSourceLists
from studentDnnModels import createFcModel
from os import mkdir, makedirs
from os.path import isdir, isfile
from keras.optimizers import Adam
from keras.models import load_model
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
    
    #==== load DNN model
    print('%s model is selected' % studentType)
    studentModel = load_model(studentFullPath)

    #==== handle save path
    tbPath = './tblogs/' + teacher + '/'
    if not isdir(tbPath):
        mkdir(tbPath)

    #==== start training
    configure(tbPath) #configure tensorboard logger
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
            #if studentType == 'CNN':
                # do something

            #==== training 
            if not checkNan(y):
                studentModel.fit(X, yScaled, batch_size=640, epochs=1, verbose=0, shuffle=True)
                results = studentModel.evaluate(X, yScaled, batch_size=19, verbose=1)
                trainLoss = results[0]
                allTrainLoss.append(trainLoss)
            else:
                print('nan found, skip this file')

        print('logging training loss...')
        log_value('training_loss', np.mean(allTrainLoss), e)
        print('saving model at epoch = %d'% e)
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
    yAll = np.ndarray((0, 3))
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
        yAll = np.concatenate((yAll, y), axis=0)
        
    yAll = np.nan_to_num(yAll)
    maxVec = np.max(yAll, axis=0)
    minVec = np.min(yAll, axis=0)
    print('%d nan files' % len(nanFiles))
    print(maxVec)
    print(minVec)
    np.savez(normParamPath, maxVec, minVec)
    return ()


'''
this function initialize an untrained student DNN model and returns its path
input:
    studentType: str, {'FC', 'CNN', 'RNN'}
    studentParentFolder: str, should be './savedStudentModels/'
output:
    studentFullPath: str, e.g., './savedStudentModels/FC/studentModel.h5'
'''
def initStudentModel(studentType, studentParentFolder):
    selectedOptimizer = Adam(lr=0.001)
    selectedLoss = 'mse'
    studentFolder = studentParentFolder + '/' + studentType + '/'
    if not isdir(studentFolder):
        makedirs(studentFolder)
    studentFullPath = studentFolder + 'studentModel.h5'
    if studentType == 'FC':
        inputDim = 1025
        studentModel = createFcModel(inputDim, selectedOptimizer, selectedLoss)
    elif studentType == 'CNN':
        studentModel = []

    print('initial untrained model saved at %s', studentFullPath)
    studentModel.save(studentFullPath)
    return studentFullPath


'''
this function is to convert the training data into CNN compatible format
input:
    X: ndarray, numBlock x numFreq STFT matrix
    y: ndarray, numBlock x 3 activation matrix
output:
    XCnn: tensor, numSample x 1 x 128 x 128 
    yCnn: tensor, numSample x 1 x 3
'''
# def prepareCnnInOut(X, y):
#     X = np.transpose(X)
#     inputMatrixMel = melspectrogram(S=X, sr=44100, n_fft=2048, hop_length=512, power=2.0, n_mels=128, fmin=0.0, fmax=20000)
#     inputTensor = np.expand_dims(inputMatrixMel, axis=0) #add a dummy dimension for sample count 1 x 128 x M
#     inputTensor = convert2dB(inputTensor) #the input is the power of mel spectrogram
#     inputTensor = scaleTensorTrackwise(inputTensor) #scale the dB scaled tensor to range of {0, 1}


#     return XCnn, yCnn


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
    normParamPath = 'normParam_' + teacher + '_genre' + str(numGenres) + '_numSongs' + str(numSongsPerGenre) +'.npz'
    newSourceTrain = []
    fileCountPerGenre = {}
    for genre in selectedGenres:
        fileCountPerGenre[genre] = 0
    for sourceFilePath, genre in sourceTrain:
        if fileCountPerGenre[genre] <= numSongsPerGenre and genre in selectedGenres:
            fileCountPerGenre[genre] += 1
            newSourceTrain.append((sourceFilePath, genre))
    return newSourceTrain, normParamPath


def main():
    allTeachers = ['pfnmf_200d', 'pfnmf_smt']
    studentParentFolder = './savedStudentModels/'
    studentType = 'FC'
    selectedGenres = ['alternative-songs', 'dance-club-play-songs', 'hot-mainstream-rock-tracks', 'latin-songs', 'pop-songs', 'r-b-hip-hop-songs']
    numSongsPerGenre = 1330
    
    #==== init model
    if CONTFLAG:
        studentFullPath = studentParentFolder + '/' + studentType + '/' + 'studentModel.h5'
        print('- continue training an existing model at %s' % studentFullPath)
    else:
        studentFullPath = initStudentModel(studentType, studentParentFolder)
        print('- start a new model for training at %s' % studentFullPath)
    
    #==== loop through all teachers
    for teacher in allTeachers:
        print('==== teacher = %s ====' % teacher)
        sourceLists = '../../preprocessData/stft_train_test_splits.npy' 
        sourceTrain, normParamPath = adjustSourceTrainList(sourceLists, teacher, selectedGenres, numSongsPerGenre)
        if not isfile(normParamPath):
            print('- loading normalization parameters for activation functions')
            estimateActivNormParam(sourceTrain, teacher, normParamPath)
        else:
            print('- normalization parameteter exists')

        print('==== start training student %s model ====' % studentType)
        trainStudent(sourceTrain, studentFullPath, studentType, teacher, normParamPath)
    return()


if __name__ == "__main__":
    print('running main() directly...')
    main()
    