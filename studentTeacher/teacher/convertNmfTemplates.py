'''
convert pre-extracted drum templates from .mat file to .npy
CW @ GTCMT 2018
'''
import numpy as np
from scipy.io import loadmat
from os.path import isdir
from os import mkdir


def convertTemplate(matPath, pyPath):
    tmp = loadmat(matPath) #HH, KD, SD
    oldWD = tmp['template']
    HH = np.expand_dims(oldWD[:, 0], axis=1)
    KD = np.expand_dims(oldWD[:, 1], axis=1)
    SD = np.expand_dims(oldWD[:, 2], axis=1)
    WD = np.concatenate((KD, SD, HH), axis=1)
    np.save(pyPath, WD)
    return ()


def main():
    templateFolder = './drumTemplates/'
    if not isdir(templateFolder):
        mkdir(templateFolder)

    #==== path to the .mat templates, the instrument order is HH, KD, SD
    temp1Path = '/Users/cw/Documents/CW_FILES/02_Github_repo/GTCMT/drumTrans_with_unlabeledData/matlab_part/templates/template_200DRUMS_2048_512.mat'
    temp2Path = '/Users/cw/Documents/CW_FILES/02_Github_repo/GTCMT/drumTrans_with_unlabeledData/matlab_part/templates/template_SMT_2048_512.mat'
    temp3Path = '/Users/cw/Documents/CW_FILES/02_Github_repo/GTCMT/drumTrans_with_unlabeledData/matlab_part/templates/template_ENST_2048_512.mat'
    
    #==== save path for the python compatible templates, the instrument order is KD, SD, HH
    temp1PyPath = templateFolder + 'template_200drums_2048_512.npy'
    temp2PyPath = templateFolder + 'template_smt_2048_512.npy'
    temp3PyPath = templateFolder + 'template_enst_2048_512.npy'

    convertTemplate(temp1Path, temp1PyPath)
    convertTemplate(temp2Path, temp2PyPath)
    convertTemplate(temp3Path, temp3PyPath)
    return()

if __name__ == "__main__":
    print('running main() directly')
    main()
