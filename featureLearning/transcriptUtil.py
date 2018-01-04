# A collection of drum transcription related functions
# CW @ GTCMT 2017

import numpy as np
from scipy.signal import argrelmax

'''
input:
    nvt: N by 1 float vector, novelty function
    order: int, window length for adaptive median filter (in samples)
    offset: float, dc offset of the threshold curve (in percentage relative to the max value)
output:
    thresCurve: N by 1 float vector, adaptive threshold based on given novelty function
'''
def medianThreshold(nvt, order, offset):
    thresCurve = np.zeros((len(nvt), 1))
    maxVal = max(nvt)

    for i in range(0, len(nvt)):
        istart = int(max([0, (i - order)]))
        iend = i + 1
        med = np.median(nvt[istart:iend])
        thresCurve[i] = offset * maxVal + med
    shift = int(round(0.5 * order))
    thresCurve[0:len(nvt)-shift] = thresCurve[shift:len(nvt)]
    return thresCurve

'''
input: 
    nvt: N by 1 float vector, novelty function
    thresCurve: N by 1 float vector, adaptive threshold curve
output:
    nvt: N by 1 float vector, thresholded novelty function
'''
def thresNvt(nvt, thresCurve):
    numBlocks = len(nvt)
    for i in range(0, numBlocks):
        if nvt[i] <= thresCurve[i]:
            nvt[i] = 0
    return nvt

'''
input: 
    nvt: N by 1 float vector, novelty function
    thresCurve: N by 1 float vector, adaptive threshold curve
    fs: int, sampling frequency
    hopSize: int, hop size of the nvt function
output:
    onsetInBlock: M by 1 int vector, onset location in block indices
    onsetInSec: M by 1 float vector, onset location in seconds
'''
def findPeaks(nvt, thresCurve, fs, hopSize):
    numBlocks = len(nvt)
    hopTime = float(hopSize)/float(fs)
    timeStamp = np.multiply(range(0, numBlocks), hopTime)
    for i in range(0, numBlocks):
        if nvt[i] <= thresCurve[i]:
            nvt[i] = 0
    order_30ms = int(round(0.03/hopTime))
    onsetInBlock,  = argrelmax(np.array(nvt), order= order_30ms)
    if len(onsetInBlock) > 0:
        onsetInSec  = timeStamp[onsetInBlock]
    else:
        onsetInSec  = []
    onsetInBlock = np.array(onsetInBlock)
    onsetInSec = np.array(onsetInSec)

    return onsetInBlock, onsetInSec

'''
input
    onsetInSec: N by 1 float vector, onset times ex. [0.125, 0.135, ...]
    length: int, the length of the output vector
    hopSize: int, the hop size of novelty function
    fs: float, sampling frequency of the signal
output
    onsetInBinary: M by 1 int vector, onset impulses ex. [0, 1, 0, ...]
'''
def onset2BinaryVector(onsetInSec, length, hopSize, fs):
    onsetInBinary = np.zeros((length, 1))
    hopTime = float(hopSize)/float(fs)
    timeStamp = np.multiply(xrange(0, length), hopTime)
    for onset in onsetInSec:
        tmp = np.subtract(timeStamp, onset)
        ind = np.argmin(abs(tmp))
        onsetInBinary[ind] = 1.0
    return onsetInBinary

'''
input
    onsets: N by 1 float vector, onset times of all drum events
    drums:  N by 1 str vector, corresponding drum names to the onset times
output
    onsets_hh: M by 1, float vector, onset times of hh
    onsets_bd: M by 1, float vector, onset times of bd
    onsets_sd: M by 1, float vector, onset times of sd
'''
def getIndividualOnset(onsets, drums):
    onsets_hh = []
    onsets_bd = []
    onsets_sd = []
    for i in range(0, len(drums)):
        if drums[i] == 'chh' or drums[i] == 'ohh':
            onsets_hh.append(onsets[i])
        elif drums[i] == 'bd':
            onsets_bd.append(onsets[i])
        elif drums[i] == 'sd':
            onsets_sd.append(onsets[i])
    onsets_hh = np.array(onsets_hh)
    onsets_bd = np.array(onsets_bd)
    onsets_sd = np.array(onsets_sd)

    return onsets_hh.ravel(), onsets_bd.ravel(), onsets_sd.ravel()


def showAllResults(resultFilePath):
    all_results = np.load(resultFilePath)
    hh_results = all_results[0]
    bd_results = all_results[1]
    sd_results = all_results[2]

    print '==== hh results ====\n'
    avg_hh = np.mean(hh_results, 0)
    print 'avg f-measure = %f, avg precision = %f, avg recall = %f\n' % (avg_hh[0], avg_hh[1], avg_hh[2])
    print '==== bd results ====\n'
    avg_bd = np.mean(bd_results, 0)
    print 'avg f-measure = %f, avg precision = %f, avg recall = %f\n' % (avg_bd[0], avg_bd[1], avg_bd[2])
    print '==== sd results ====\n'
    avg_sd = np.mean(sd_results, 0)
    print 'avg f-measure = %f, avg precision = %f, avg recall = %f\n' % (avg_sd[0], avg_sd[1], avg_sd[2])
    print '==== average across instruments ====\n'
    print 'avg avg f-measure = %f\n' % np.mean([avg_hh[0], avg_bd[0], avg_sd[0]])
    print 'avg avg precision = %f\n' % np.mean([avg_hh[1], avg_bd[1], avg_sd[1]])
    print 'avg avg recall    = %f\n' % np.mean([avg_hh[2], avg_bd[2], avg_sd[2]])

    return









