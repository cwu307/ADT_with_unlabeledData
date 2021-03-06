'''
unsupervised feature learning using convolutional autoencoder
CW @ GTCMT 2017
'''

import numpy as np
import shutil
import time
import sys
sys.path.insert(0, '../featureExtraction')
from extractFeatures import prepareConvnetInput
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from dnnModels import createAeModel
from FileUtil import convert2dB, scaleTensorTrackwise, reshapeInputTensor, invReshapeInputTensor
from librosa.feature import melspectrogram
from tensorboard_logger import configure, log_value
from os.path import isdir
from os import mkdir
EMBEDDIM = 4
CONTINUEFLAG = True

def trainAeModel(sourceLists, targetLists, modelSavePath, tbpath):
    #==== define data path
    check_path = modelSavePath + 'checkpoint.h5'
    ae_path = modelSavePath + 'ae.h5'
    ext1Path = modelSavePath + 'ext1.h5'
    ext2Path = modelSavePath + 'ext2.h5'
    ext3Path = modelSavePath + 'ext3.h5'
    ext4Path = modelSavePath + 'ext4.h5'
    ext5Path = modelSavePath + 'ext5.h5'
    #shutil.rmtree(tbpath)
    configure(tbpath) #configure tensorboard logger

    #==== load all train-test split lists
    sourceAllLists = np.load(sourceLists)
    targetAllLists = np.load(targetLists)
    sourceTrain = sourceAllLists[0]
    sourceVal = sourceAllLists[1]
    sourceTest = sourceAllLists[2]
    targetTrain = targetAllLists[0]
    targetVal = targetAllLists[1]
    targetTest = targetAllLists[2]

    #==== define DNN parameters
    inputDim = 128
    inputDim2 = 128
    embedDim = EMBEDDIM
    numEpochs = 10
    selectedOptimizer = Adam(lr=0.001)
    selectedLoss = 'mse'
    checker = ModelCheckpoint(check_path)
    if CONTINUEFLAG:
        ae, ext1, ext2, ext3, ext4, ext5 = contineTraining(modelSavePath)
    else:
        ae, ext1, ext2, ext3, ext4, ext5 = createAeModel(inputDim, inputDim2, embedDim, selectedOptimizer, selectedLoss)

    for e in range(0, numEpochs):
        print("==== epoch %d ====" % e)
        print('looping through %d training data:'% len(sourceTrain))
        allTrainLoss =[]
        for i in range(0, len(sourceTrain)):
            print('file number %d' % i)
            sourceFilepath, sourceLabel = sourceTrain[i]
            targetFilepath, targetLabel = targetTrain[i]
            source, target = prepareData(sourceFilepath, targetFilepath)
            ae.fit(source, target, epochs=1, batch_size=19, callbacks=[checker], verbose=0, shuffle=False)
            results = ae.evaluate(source, target, batch_size=19, verbose=1)
            trainLoss = results[0]
            allTrainLoss.append(trainLoss)
        
        print('evaluating validation loss')
        allValLoss = []
        for i in range(0, len(sourceVal)):
            sourceFilepath, sourceLabel = sourceVal[i]
            targetFilepath, targetLabel = targetVal[i]
            source, target = prepareData(sourceFilepath, targetFilepath)
            results = ae.evaluate(source, target, batch_size=19, verbose=1)
            valLoss = results[0]
            allValLoss.append(valLoss)
        print('logging training loss...')
        log_value('training_loss', np.mean(allTrainLoss), e)
        log_value('validation_loss', np.mean(allValLoss), e)

        print('save temporary results of %d epoch' % e)
        ae.save(ae_path)
        ext1.save(ext1Path)
        ext2.save(ext2Path)
        ext3.save(ext3Path)
        ext4.save(ext4Path)
        ext5.save(ext5Path)


def contineTraining(modelSavePath):
    ae_path = modelSavePath + 'ae.h5'
    ext1Path = modelSavePath + 'ext1.h5'
    ext2Path = modelSavePath + 'ext2.h5'
    ext3Path = modelSavePath + 'ext3.h5'
    ext4Path = modelSavePath + 'ext4.h5'
    ext5Path = modelSavePath + 'ext5.h5'
    ae = load_model(ae_path)
    ext1 = load_model(ext1Path)
    ext2 = load_model(ext2Path)
    ext3 = load_model(ext3Path)
    ext4 = load_model(ext4Path)
    ext5 = load_model(ext5Path)
    return ae, ext1, ext2, ext3, ext4, ext5



def getRandomWeightAeModel(modelSavePath):
    if not isdir(modelSavePath):
        mkdir(modelSavePath)
    #==== define data path
    ae_path = modelSavePath + 'ae.h5'
    ext1Path = modelSavePath + 'ext1.h5'
    ext2Path = modelSavePath + 'ext2.h5'
    ext3Path = modelSavePath + 'ext3.h5'
    ext4Path = modelSavePath + 'ext4.h5'
    ext5Path = modelSavePath + 'ext5.h5'

    #==== define DNN parameters
    inputDim = 128
    inputDim2 = 128
    embedDim = EMBEDDIM
    selectedOptimizer = Adam(lr=0.1)
    selectedLoss = 'mse'
    ae, ext1, ext2, ext3, ext4, ext5 = createAeModel(inputDim, inputDim2, embedDim, selectedOptimizer, selectedLoss)

    print('saving trained models...')
    ae.save(ae_path)
    ext1.save(ext1Path)
    ext2.save(ext2Path)
    ext3.save(ext3Path)
    ext4.save(ext4Path)
    ext5.save(ext5Path)


def prepareData(sourceFilePath, targetFilePath):
    sourceFilepathAdjusted = '../../preprocessData' + sourceFilePath[1:]
    targetFilepathAdjusted = '../../preprocessData' + targetFilePath[1:]
    source = np.load(sourceFilepathAdjusted) 
    target = np.load(targetFilepathAdjusted)
    source = prepareConvnetInput(source)
    target = prepareConvnetInput(target)
    return source, target

def main():
    stftLists = '../../preprocessData/stft_train_test_splits.npy'
    stftPLists = '../../preprocessData/stft_p_train_test_splits.npy'

    #==== AE
    print('Getting autoencoder models')
    modelSavePath = './savedAeModels/'
    tbpath = './tblogs/ae_run'
    trainAeModel(sourceLists=stftLists, targetLists=stftLists, modelSavePath=modelSavePath, tbpath=tbpath)

    #==== DAE
    # print('Getting denoising autoencoder models')
    # modelSavePath = './savedDaeModels/'
    # tbpath = './tblogs/dae_run'   
    # trainAeModel(sourceLists=stftLists, targetLists=stftPLists, modelSavePath=modelSavePath, tbpath=tbpath)

    #==== Random Weights
    # print('Getting models with random weights')
    # modelSavePath = './savedRandomAeModels/'
    # getRandomWeightAeModel(modelSavePath=modelSavePath)
    return()

if __name__ == "__main__":
    print('running main() directly')
    main()