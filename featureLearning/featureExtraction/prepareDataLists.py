'''
this script is used to generate file_lists for multiple drum datasets
CW @ GTCMT 2017
'''
import numpy as np
from glob import glob
from os import listdir, mkdir
from os.path import isdir

def getEnstDataList(dataDir):
    enstList = []
    folders = ['drummer1', 'drummer2', 'drummer3']
    for folder in folders:
        audioPath = dataDir + folder + '/audio/'
        annPath = dataDir + folder + '/annotation/'
        audioList = sorted(glob((audioPath +  '*.wav')))
        annList = sorted(glob(annPath + '*.txt'))
        for i in range(0, len(audioList)):
            audioFilename = audioList[i].split('/')[-1].split('_')[0]
            annFilename = annList[i].split('/')[-1].split('_')[0]
            assert(audioFilename == annFilename)
            enstList.append((audioList[i], annList[i]))
    return enstList

def getRbmaDataList(dataDir):
    rbmaList = []
    audioPath = dataDir + 'audio/'
    annPath = dataDir + 'annotations/'
    audioList = sorted(glob(audioPath + '*.wav'))
    annList = sorted(glob(annPath + '*.txt'))
    for i in range(0, len(audioList)):
        audioFilename = audioList[i].split('/')[-1].split('.')[0]
        annFilename = annList[i].split('/')[-1].split('.')[0]
        assert(audioFilename == annFilename)
        rbmaList.append((audioList[i], annList[i]))
    return rbmaList

def getMdbDataList(dataDir):
    mdbList = []
    audioPath = dataDir + 'audio/full_mix/'
    annPath = dataDir + 'annotations/class/'
    audioList = sorted(glob(audioPath + '*.wav'))
    annList = sorted(glob(annPath + '*.txt'))
    for i in range(0, len(audioList)):
        audioFilename = audioList[i].split('/')[-1].split('_MIX')[0]
        annFilename = annList[i].split('/')[-1].split('_class')[0]
        assert(audioFilename == annFilename)
        mdbList.append((audioList[i], annList[i]))
    return mdbList

def get2005List(dataDir):
    m2005List = []
    audioPath = dataDir + 'audio/'
    annPath = dataDir + 'annotations/'
    audioList = sorted(glob(audioPath + '*.wav'))
    annList = sorted(glob(annPath + '*.txt'))
    for i in range(0, len(audioList)):
        audioFilename = audioList[i].split('/')[-1].split('.')[0]
        annFilename = annList[i].split('/')[-1].split('.')[0]
        assert(audioFilename == annFilename)
        m2005List.append((audioList[i], annList[i]))
    return m2005List

def displayList(dataList):
    for item in dataList:
        print(item)
    return()


def main_server():
    if not isdir('./dataLists/'):
        mkdir('./dataLists/')

    enstDir = '/data/labeled_drum_datasets/CW_ENST_minus_one_wet_new_ratio/'
    enstList = getEnstDataList(enstDir)
    displayList(enstList)
    np.save('./dataLists/enstList.npy', enstList)

    rbmaDir = '/data/labeled_drum_datasets/2017_mirex/RBMA/'
    rbmaList = getRbmaDataList(rbmaDir)
    displayList(rbmaList)
    np.save('./dataLists/rbmaList.npy', rbmaList)

    m2005Dir = '/data/labeled_drum_datasets/2017_mirex/2005/'
    m2005List = get2005List(m2005Dir)
    displayList(m2005List)
    np.save('./dataLists/m2005List.npy', m2005List)

    mdbDir = '/data/labeled_drum_datasets/MDB Drums/'
    mdblist = getMdbDataList(mdbDir)
    displayList(mdblist)
    np.save('./dataLists/mdbList.npy', mdblist)
    return()

def main_local():
    if not isdir('./dataLists/'):
        mkdir('./dataLists/')

    enstDir = '/Volumes/CW_MBP15/Datasets/labeled_drum_datasets/CW_ENST_minus_one_wet_new_ratio/'
    enstList = getEnstDataList(enstDir)
    displayList(enstList)
    np.save('./dataLists/enstList.npy', enstList)

    rbmaDir = '/Volumes/CW_MBP15/Datasets/labeled_drum_datasets/2017_mirex/RBMA/'
    rbmaList = getRbmaDataList(rbmaDir)
    displayList(rbmaList)
    np.save('./dataLists/rbmaList.npy', rbmaList)

    m2005Dir = '/Volumes/CW_MBP15/Datasets/labeled_drum_datasets/2017_mirex/2005/'
    m2005List = get2005List(m2005Dir)
    displayList(m2005List)
    np.save('./dataLists/m2005List.npy', m2005List)

    mdbDir = '/Volumes/CW_MBP15/Datasets/labeled_drum_datasets/MDB Drums/'
    mdblist = getMdbDataList(mdbDir)
    displayList(mdblist)
    np.save('./dataLists/mdbList.npy', mdblist)
    return()

if __name__ == "__main__":
    print('running main() directly')
    main_server()




