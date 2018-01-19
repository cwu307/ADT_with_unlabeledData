'''
this is the re-implementation of the PFNMF from https://github.com/cwu307/NmfDrumToolbox
CW @ GTCMT 2018
'''
import numpy as np

REALMIN = 10e-6#np.finfo(float).tiny
HOPSIZE = 512
WINSIZE = 2048
FS = 44100.0
'''
partially fixed nonnegative matrix factorization
input:
    X: float, numFreq by numBlock matrix, input magnitude spectrogram
    WD: float, numFreqD by rd matrix, drum dictionary, rd is the rank of drum components (usually rd = 3 for KD, SD, HH)
    HD: float, rd by numBlock matrix, drum activation matrix
    WH: float, numFreqH by rh matrix, harmonic dictionary, rh is the rank of harmonic components
    HH: float, rh by numBlock matrix, harmonic activation matrix
    rh: int, rank of harmonic matrix (usually set to 50)

output:
    WD: float, numFreqD by rd matrix, updated drum dictionary
    HD: float, rd by numBlock, updated drum activation matrix
    WH: float, numFreqH by rh matrix, updated harmonic dictionary
    HH: float, rh by numBlock, updated harmonic activation matrix
    err: float, numIter by 1 vector, the error curve over iterations
'''
def pfNmf(X, WD, HD, WH, HH, rh, sparsity):
    X = X + REALMIN
    numFreqX, numBlock = np.shape(X)
    numFreqD, rd = np.shape(WD)

    #==== initialization
    WD_update = 0
    HD_update = 0
    WH_update = 0
    HH_update = 0 

    if len(WH) != 0:
        numFreqH, rh = np.shape(WH)
    else:
        WH = np.random.rand(numFreqD, rh)
        numFreqH, dump = np.shape(WH)
        WH_update = 1
    
    assert(numFreqD == numFreqX)
    assert(numFreqH == numFreqX)

    if len(HD) != 0:
        WD_update = 1
    else:
        HD = np.random.rand(rd, numBlock)
        HD_update = 1

    if len(HH) == 0:
        HH = np.random.rand(rh, numBlock)
        HH_update = 1
    
    alpha = float(rh + rd) / rd
    beta  = rh / float(rh + rd)

    #==== normalize W & H matrix (normalized by L1 norm)
    for i in range(0, rd):
        WD[:, i] = np.divide(WD[:, i], np.linalg.norm(WD[:, i], 1))
    
    for i in range(0, rh):
        WH[:, i] = np.divide(WH[:, i], np.linalg.norm(WH[:, i], 1))

    count = 0
    err = []
    rep = np.ones((numFreqX, numBlock)) 

    #==== start iteration
    while (count < 300):

        approx = alpha * np.dot(WD, HD) + beta * np.dot(WH, HH)

        #==== update
        if HD_update:
            HD = np.multiply(HD, np.divide(np.dot(np.transpose(alpha * WD), np.divide(X, approx)), (np.dot(np.transpose(alpha * WD), rep) + sparsity)))
        
        if HH_update:
            HH = np.multiply(HH, np.divide(np.dot(np.transpose(beta * WH), np.divide(X, approx)), (np.dot(np.transpose(beta * WH), rep))))
        
        if WD_update:
            WD = np.multiply(WD, np.divide(np.dot(np.divide(X, approx), np.transpose(alpha * HD)),  np.dot(rep, np.transpose(alpha * HD))))
        
        if WH_update:
            WH = np.multiply(WH, np.divide(np.dot(np.divide(X, approx), np.transpose(beta * HH)),  np.dot(rep, np.transpose(beta * HH))))

        #==== normalize W & H 
        for i in range(0, rd):
            WD[:, i] = np.divide(WD[:, i], np.linalg.norm(WD[:, i], 1))
    
        for i in range(0, rh):
            WH[:, i] = np.divide(WH[:, i], np.linalg.norm(WH[:, i], 1))

        #==== compute loss  
        curErr = klDivergence(X, (alpha * np.dot(WD, HD) + beta * np.dot(WH, HH))) + sparsity * np.sum(np.linalg.norm(HD, 1, axis=1))
        err.append(curErr)
        
        if count >= 1:
            if abs(err[count] - err[count - 1]) / (err[0] - err[count] + REALMIN) < 0.001:
                break
        count += 1
    return WD, HD, WH, HH, err


'''
compute generalized KL divergence
https://en.wikipedia.org/wiki/Bregman_divergence
input:
    p: float, m by n matrix
    q: float, m by n matrix
output: 
    D: float, scalar, KL divergence
'''
def klDivergence(p, q):
    pp = np.log(p + REALMIN)
    qq = np.log(q + REALMIN)
    D = np.sum(np.multiply(pp, (pp - qq)) - pp + qq)
    return D

def main():
    # quick test of the implemented function
    from librosa import load
    from librosa.core import stft

    audioPath = 'test_audio.wav'
    y, sr = load(audioPath, sr=FS, mono=True)
    X = stft(y, n_fft=WINSIZE, hop_length=HOPSIZE, win_length=WINSIZE, window='hann', center=True)
    X = abs(X)
    WD = np.load('./drumTemplates/template_enst_2048_512.npy')
    WD, HD, WH, HH, err = pfNmf(X, WD, HD=[], WH=[], HH=[], rh=50, sparsity=0.0)
    return()

def debug():
    from librosa import load
    from librosa.core import stft
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, '../../featureLearning/')
    from transcriptUtil import thresNvt, medianThreshold
    ORDER = round(0.1 / (HOPSIZE / FS))
    OFFSET = 0.12


    dataListPath = '../../featureLearning/featureExtraction/dataLists/m2005List.npy'
    dataList = np.load(dataListPath) 
    audioPath, annPath = dataList[0]
    y, sr = load(audioPath, sr=FS, mono=True)
    X = stft(y, n_fft=WINSIZE, hop_length=HOPSIZE, win_length=WINSIZE, window='hann', center=True)
    X = abs(X)
    WD = np.load('./drumTemplates/template_200drums_2048_512.npy')
    WD, HD, WH, HH, err = pfNmf(X, WD, HD=[], WH=[], HH=[], rh=50, sparsity=0.0)
    nvt = HD[0, :]
    nvt = np.divide(nvt, np.max(abs(nvt)))
    thresCurve = medianThreshold(nvt, ORDER, OFFSET)
    nvtNew = thresNvt(nvt, thresCurve)
    plt.subplot(211)
    plt.plot(nvt)
    plt.subplot(212)
    plt.plot(nvtNew)
    plt.savefig('test_200d_bd.png', format='png')
    return


if __name__ == "__main__":
    print('running main() directly')
    #main()
    debug()