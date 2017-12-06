'''
this script is to generate evaluation results!
CW @ GTCMT 2017
'''
import numpy as np
from mir_eval.onset import f_measure
import sys
sys.path.insert(0, '../featureLearning/featureExtraction')




def evaluateEntireFolder(predictionListPath, methodOption):
    #loop through all files in the folder
    #record p, r, f per instrument per track
    #compute averages 
    #write txt file 
    predictionList = np.load(predictionListPath)
    for annPath, predPath in predictionList:
        predPathMod = '../' + methodOption + predPath[1:]
        resultPerInst = evaluateSingleTrack(annPath, predPathMod)
        print(resultPerInst)

    return()

'''
input:
    annPath: str, path to the ground truth annotation .txt file
    predPath: str, path to the prediction .txt file
output:
    resultPerInst: list of tuples [(bd_f, bd_p, bd_r), (sd_f, sd_p, sd_r), (hh_f, hh_p, hh_r)]
'''
def evaluateSingleTrack(annPath, predPath):
    allDrums = [0, 1, 2]
    resultPerInst = []
    for i in range(0, len(allDrums)):
        targetDrumNum = allDrums[i]
        refOnsets = getTargetDrumOnsets(annPath, targetDrumNum)
        estOnsets = getTargetDrumOnsets(predPath, targetDrumNum)
        window = 0.05
        f, p, r = f_measure(refOnsets, estOnsets, window)
        resultPerInst.append((f, p, r))
    return resultPerInst

'''
input:
    annPath
    targetDrumNum:
        0:bd
        1:sd
        2:hh
        3:others
output:
    targetDrumOnsets: onset time (sec) of the target drum
'''
def getTargetDrumOnsets(annPath, targetDrumNum):
    targetDrumOnsets = []
    annFile = open(annPath, 'r')
    for line in annFile.readlines():
        onset, drum = line.split()
        if drum == 'KD' or drum == 'bd' or drum == '0':
            classNum = 0
        elif drum == 'SD' or drum == 'sd' or drum == '1':
            classNum = 1
        elif drum == 'HH' or drum == 'chh' or drum == 'ohh' or drum == '2':
            classNum = 2
        else:
            classNum = 3
        if classNum == targetDrumNum:
            targetDrumOnsets.append(float(onset))
    return np.asarray(targetDrumOnsets)

'''
input:
    methodOption: str, viable options are 'featurelearning' and 'studentTeacher' (to be added)
    heldOutOption: str, viable options are 'enst', 'mdb', 'rbma', 'm2005'
    featureOption: str, vialbe options are 'baseline', 'convRandom', 'convAe', 'convDae'
output:
    predictionListPath: str, the path to the predictionList.npy files
'''
def getPredictionListPath(methodOption, heldOutOption, featureOption):
    predictionListPath = '../' + methodOption + '/predictionResults/' + heldOutOption + '_feat_' + featureOption + '/' + heldOutOption + '_feat_' + featureOption + '_predictionList.npy' 
    return predictionListPath

def main():
    methodOption = 'featureLearning'
    heldOutOption = 'enst'
    featureOption = 'baseline'
    predictionListPath = getPredictionListPath(methodOption, heldOutOption, featureOption)
    evaluateEntireFolder(predictionListPath, methodOption)
    return()


if __name__ == "__main__":
    print('running main() directly...')
    main()
