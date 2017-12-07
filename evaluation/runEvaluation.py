'''
this script is to generate evaluation results!
CW @ GTCMT 2017
'''
import numpy as np
from mir_eval.onset import f_measure

def evaluateEntireFolder(predictionListPath, methodOption):
    #loop through all files in the folder
    #record p, r, f per instrument per track
    #compute averages 
    #write txt file
    filename = predictionListPath.split('/')[-1][0:-3]
    resultsTxtPath = './evaluationResults/' + methodOption + '_' + filename + 'txt' 
    predictionList = np.load(predictionListPath)
    allResultsPerInst = []
    for annPath, predPath in predictionList:
        predPathMod = '../' + methodOption + predPath[1:]
        resultsPerInst = evaluateSingleTrack(annPath, predPathMod)
        allResultsPerInst.append((predPathMod, resultsPerInst))
    writeResults2TxtFile(resultsTxtPath, allResultsPerInst)
    return()

def writeResults2TxtFile(resultsTxtPath, allResultsPerInst):
    resultsTxt = open(resultsTxtPath, 'w')
    resultsTxt.write('songID    all_f    all_p    all_r    bd_f    bd_p    bd_r    sd_f    sd_p    sd_r    hh_f    hh_p    hh_r\n')
    fListAll = []
    pListAll = []
    rListAll = []
    fListBd = []
    pListBd = []
    rListBd = []
    fListSd = []
    pListSd = []
    rListSd = []
    fListHh = []
    pListHh = []
    rListHh = []
    c = 0
    for predPath, resultsPerInst in allResultsPerInst:
        c += 1
        bd_f, bd_p, bd_r = resultsPerInst[0]
        sd_f, sd_p, sd_r = resultsPerInst[1]
        hh_f, hh_p, hh_r = resultsPerInst[2]
        all_f = np.mean([bd_f, sd_f, hh_f])
        all_p = np.mean([bd_p, sd_p, hh_p])
        all_r = np.mean([bd_r, sd_r, hh_r])
        resultsTxt.write('%s    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f\n' % (str(c), all_f, all_p, all_r,
                         bd_f, bd_p, bd_r, sd_f, sd_p, sd_r, hh_f, hh_p, hh_r))
        fListAll.append(all_f)
        pListAll.append(all_p)
        rListAll.append(all_r)
        fListBd.append(bd_f)
        pListBd.append(bd_p)
        rListBd.append(bd_r)
        fListSd.append(sd_f)
        pListSd.append(sd_p)
        rListSd.append(sd_r)
        fListHh.append(hh_f)
        pListHh.append(hh_p)
        rListHh.append(hh_r)
    resultsTxt.write('average    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f    %.2f\n' % (np.mean(fListAll), np.mean(pListAll),
                        np.mean(rListAll), np.mean(fListBd), np.mean(pListBd), np.mean(rListBd), np.mean(fListSd), np.mean(pListSd), np.mean(rListSd), 
                        np.mean(fListHh), np.mean(pListHh), np.mean(rListHh)))
    resultsTxt.close()
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
        if len(refOnsets) == 0:
            print('%s \n ==== ref has no %d drum' % (annPath, targetDrumNum))
        estOnsets = getTargetDrumOnsets(predPath, targetDrumNum)
        if len(estOnsets) == 0:
            print('%s \n ==== est has no %d drum' % (predPath, targetDrumNum))
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
    allHeldOutOptions = ['enst', 'mdb', 'rbma', 'm2005']
    allFeatureOptions = ['baseline', 'convRandom']#, 'convAe', 'convDae']

    for heldOutOption in allHeldOutOptions:
        for featureOption in allFeatureOptions:
            # heldOutOption = 'enst'
            # featureOption = 'convRandom'
            predictionListPath = getPredictionListPath(methodOption, heldOutOption, featureOption)
            evaluateEntireFolder(predictionListPath, methodOption)
    return()


if __name__ == "__main__":
    print('running main() directly...')
    main()
