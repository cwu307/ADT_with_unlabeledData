'''
this script is to preprocess the unlabeled dataset using the following steps:
1) select 1900 songs per genre, 6 genres
2) export the selected entries as .txt
3) for each song, make sure the fs = 44100, downmixed to monophonic
4) take 30 secs (from middle)
5) compute STFT
6) compute HPSS, get STFT of percussive part
CW @ GTCMT 2017
'''
import numpy as np
from os import listdir, mkdir
from os.path import isdir, isfile
from librosa.core import load, stft
from librosa.decompose import hpss

'''
genre_paths = getGenrePaths(UNLABELED_DATA_DIR)
input
    str, path to the unlabeled drum dataset audio folder (the folder that contains subfolders of genres)
output
    list tuples, each tuple contains: (genre, genre_path)
'''
def getGenres(UNLABELED_DATA_DIR):
    genre_info = []
    raw_list = listdir(UNLABELED_DATA_DIR)
    for genre in raw_list:
        if genre[0] != '.':
            single_genre_path = UNLABELED_DATA_DIR + genre + '/'
            genre_info.append((genre, single_genre_path))
    return genre_info

'''
input:
    parentFolder: string, directory to the parent folder
    ext: string, extension name of the interested files
output:
    filePathList: list, directory to the files
'''
def getFilePathList(folderpath, ext):
    allfiles = listdir(folderpath)
    for item in allfiles:
        if item[0] == '.':
            allfiles.remove(item)
    allfiles = sorted(allfiles, key=lambda f: int(f.split("_")[0]))
    filePathList = []
    filenames = []
    for filename in allfiles:
        if ext in filename:
            filepath = folderpath + filename
            filePathList.append(filepath)
            filenames.append(filename)
    return filePathList, filenames

'''
y_segment = getSegmentFromSong(y, norm_flag, start_loc, duration)
input
    y: float ndarray, numSamples by 1
    norm_flag: bool, normalize by maximum value or not
    start_loc: str, 'beginning' or 'middle' (display warnning if it goes out of boundary)
    duration: float, duration of the segment in seconds
output
    y_segment: float ndarray, round(duration * sr) by 1 
'''
def getSegmentFromSong(y, sr, norm_flag, start_loc, duration):
    duration_in_samples = round(duration * sr)
    if norm_flag:
        print('normalize the file by the maximum value')
        y = np.divide(y, np.max(abs(y)))
    if start_loc == 'beginning':
        istart = 0
        iend = istart + duration_in_samples
    elif start_loc == 'middle':
        istart = round(0.5 * len(y)) 
        iend = istart + duration_in_samples
    if iend > len(y):
        print('the original file is shorter than requested duration. Change starting point and try again')
        istart = 0
        iend = istart + duration_in_samples
    if iend > len(y):
        print('the duration is still not enough. Zero padding')
        istart = 0
        iend = istart + duration_in_samples
        gap = iend - len(y)
        zeropad = np.zeros((gap, 1))
        y = np.concatenate((y, zeropad), axis=0)
    y_segment = y[int(istart):int(iend)]
    return y_segment

'''
check 
'''
def checkSaveRepos(save_repo):
    if not isdir(save_repo):
        print('%s does not exist, making new directory...' % save_repo)
        mkdir(save_repo)
    return(save_repo)


def preprocUnlabeledDataset(UNLABELED_DATA_DIR, METADATA_DIR):
    genre_info = getGenres(UNLABELED_DATA_DIR)
    print('checking stft save repositories...')
    save_stft_repo = checkSaveRepos('./stft/')
    save_stft_p_repo = checkSaveRepos('./stft_p/')
    sr = 44100
    window_size = 2048
    hop_size = 512
    ext_type = 'mp3'
    tmp = np.load(METADATA_DIR)
    clean_list = tmp[0]
    problem_list = tmp[1]

    for path, genre in clean_list:
        print('checking genre repositories... %s' % genre)
        save_genre_repo = checkSaveRepos(save_stft_repo + genre + '/')
        save_genre_p_repo = checkSaveRepos(save_stft_p_repo + genre + '/') 
      
        tmp = path.split('/')
        file_name = tmp[3]
        song_name = file_name.split('.')[0]
        song_path = UNLABELED_DATA_DIR + genre + '/' + file_name
        stft_file_path = save_genre_repo + song_name + '.npy'
        stft_p_file_path = save_genre_p_repo + song_name + '.npy'

        if isfile(stft_file_path) and isfile(stft_p_file_path):
            print('file already exist! go to next song')
        else:    
            print('processing song: %s' % song_name)
            y, sr = load(song_path, sr=sr) #this function takes care of both sr and mono
            y_segment = getSegmentFromSong(y, sr, True, 'middle', 29)
            #print(y_segment)
            print('computing STFT and HPSS')
            Y = stft(y_segment, n_fft=window_size, hop_length=hop_size, window='hann')
            Y_mag = abs(Y)
            H, P = hpss(Y_mag, margin=1.0) #Y = H + P

            print('saving results...')
            np.save(stft_file_path, Y_mag)
            np.save(stft_p_file_path, P)
    return ()


def main():
    UNLABELED_DATA_DIR = '/home/../../data/unlabeledDrumDataset/audio/'
    METADATA_DIR = './unlabeled_data_cleanup_metadata.npy'
    preprocUnlabeledDataset(UNLABELED_DATA_DIR, METADATA_DIR)
    print('Finished')
    return()

if __name__ == "__main__":
    print('main is running directly')
    main()

