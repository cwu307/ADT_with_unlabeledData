'''
unsupervised feature learning using convolutional autoencoder
CW @ GTCMT 2017
'''

import numpy as np
import shutil
import time
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from dnnModels import createAeModel
from FileUtil import convert2dB, scaleTensorTrackwise, reshapeInputTensor, invReshapeInputTensor
from librosa.feature import melspectrogram
from tensorboard_logger import configure, log_value
from os.path import isdir
from os import mkdir
preprocessingFlag = True

def trainAeModel(sourceLists, targetLists, modelSavePath, tbpath):
    #==== define data path
    check_path = modelSavePath + 'checkpoint.h5'
    ae_path = modelSavePath + 'ae.h5'
    ext1Path = modelSavePath + 'ext1.h5'
    ext2Path = modelSavePath + 'ext2.h5'
    ext3Path = modelSavePath + 'ext3.h5'
    ext4Path = modelSavePath + 'ext4.h5'
    ext5Path = modelSavePath + 'ext5.h5'
    shutil.rmtree(tbpath)
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
    embedDim = 8
    numEpochs = 30
    selectedOptimizer = Adam(lr=0.001)
    selectedLoss = 'mse'
    checker = ModelCheckpoint(check_path)
    ae, ext1, ext2, ext3, ext4, ext5 = createAeModel(inputDim, inputDim2, embedDim, selectedOptimizer, selectedLoss)

    for e in range(0, numEpochs):
        print("==== epoch %d ====" % e)
        print('looping through %d training data:'% len(sourceTrain))
        allTrainLoss =[]
        tic = time.time()
        for i in range(0, len(sourceTrain)):
            if i % 10 == 0:
                print(i)
                # do stuff
                print(time.time() - tic)
            sourceFilepath, sourceLabel = sourceTrain[i]
            targetFilepath, targetLabel = targetTrain[i]
            source, target = prepareData(sourceFilepath, targetFilepath)
            ae.fit(source, target, epochs=1, batch_size=19, callbacks=[checker], verbose=0, shuffle=False)
            results = ae.evaluate(source, target, batch_size=19, verbose=0)
            trainLoss = results[0]
            allTrainLoss.append(trainLoss)
        
        print('evaluating validation loss')
        allValLoss = []
        for i in range(0, len(sourceVal)):
            sourceFilepath, sourceLabel = sourceVal[i]
            targetFilepath, targetLabel = targetVal[i]
            source, target = prepareData(sourceFilepath, targetFilepath)
            results = ae.evaluate(source, target, batch_size=19, verbose=0)
            valLoss = results[0]
            allValLoss.append(valLoss)
        print('logging training loss...')
        log_value('training_loss', np.mean(allTrainLoss), e)
        log_value('validation_loss', np.mean(allValLoss), e)

    print('saving trained models...')
    ae.save(ae_path)
    ext1.save(ext1Path)
    ext2.save(ext2Path)
    ext3.save(ext3Path)
    ext4.save(ext4Path)
    ext5.save(ext5Path)

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
    embedDim = 8
    selectedOptimizer = Adam(lr=0.001)
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
    source = melspectrogram(S=source, sr=44100, n_fft=2048, hop_length=512, power=2.0, n_mels=128, fmin=0.0, fmax=20000)
    target = melspectrogram(S=target, sr=44100, n_fft=2048, hop_length=512, power=2.0, n_mels=128, fmin=0.0, fmax=20000)
    source = np.expand_dims(source, axis=0)
    target = np.expand_dims(target, axis=0)
    if preprocessingFlag:
        source = convert2dB(source)
        target = convert2dB(target)
    source = scaleTensorTrackwise(source)
    target = scaleTensorTrackwise(target)
    source = reshapeInputTensor(source)
    target = reshapeInputTensor(target)
    source = np.expand_dims(source, axis=1) #add batch dimension 1 x 1 x dim1 x dim2
    target = np.expand_dims(target, axis=1)
    return source, target

def main():
    # stftLists = '../../preprocessData/stft_train_test_splits.npy'
    # stftPLists = '../../preprocessData/stft_p_train_test_splits.npy'

    #==== AE
    # print('Getting autoencoder models')
    # modelSavePath = './savedAeModels/'
    # tbpath = './tblogs/ae_run'
    # trainAeModel(sourceLists=stftLists, targetLists=stftLists, modelSavePath=modelSavePath, tbpath=tbpath)

    #==== DAE
    # print('Getting denoising autoencoder models')
    # modelSavePath = './savedDaeModels/'
    # tbpath = './tblogs/dae_run'   
    # trainAeModel(sourceLists=stftLists, targetLists=stftPLists, modelSavePath=modelSavePath, tbpath=tbpath)

    #==== Random Weights
    print('Getting models with random weights')
    modelSavePath = './savedRandomAeModels/'
    getRandomWeightAeModel(modelSavePath=modelSavePath)
    return()

if __name__ == "__main__":
    print('running main() directly')
    main()