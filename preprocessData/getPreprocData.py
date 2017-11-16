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
from os.path import isdir
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
        y = np.divide(y, np.max(abs(y)))
    if start_loc == 'beginning':
        istart = 0
        iend = istart + duration_in_samples
    elif start_loc == 'middle':
        istart = round(0.5 * len(y)) 
        iend = istart + duration_in_samples
    if iend > len(y):
        print('the original file is shorter than requested duration, zero padding')
        y = np.concatenate((y, np.zeros(len(y) - iend + 1, 1)), axis=0)
    y_segment = y[int(istart):int(iend)]
    return y_segment

def checkSaveRepos(save_repo):
    if not isdir(save_repo):
        print('%s does not exist, making new directory...' % save_repo)
        mkdir(save_repo)
    return(save_repo)


def preprocUnlabeledDataset(UNLABELED_DATA_DIR, sr, window_size, hop_size, num_song_per_genre, ext_type):
    genre_info = getGenres(UNLABELED_DATA_DIR)
    print('checking stft save repositories...')
    save_stft_repo = checkSaveRepos('./stft/')
    save_stft_p_repo = checkSaveRepos('./stft_p/')
    
    for genre, genre_path in genre_info:
        print('checking genre repositories...')
        save_genre_repo = checkSaveRepos(save_stft_repo + genre + '/')
        save_genre_p_repo = checkSaveRepos(save_stft_p_repo + genre + '/')
        song_list, songnames = getFilePathList(genre_path, ext_type)

        for i in range(0, num_song_per_genre):
            song_path = song_list[i]
            songname = songnames[i]
            print('processing song: %s' % songname)
            y, sr = load(song_path, sr=sr) #this function takes care of both sr and mono
            y_segment = getSegmentFromSong(y, sr, True, 'middle', 30)

            print('computing STFT and HPSS')
            Y = stft(y_segment, n_fft=window_size, hop_length=hop_size, window='hann')
            Y_mag = abs(Y)
            H, P = hpss(Y_mag, margin=1.0) #Y = H + P

            print('saving results...')
            stft_file_path = save_genre_repo + songname + '.npy'
            stft_p_file_path = save_genre_p_repo + songname + '.npy'
            np.save(stft_file_path, Y_mag)
            np.save(stft_p_file_path, P)
    return ()


def main():
    UNLABELED_DATA_DIR = '/home/../../data/unlabeledDrumDataset/audio/'
    sr = 44100
    window_size = 2048
    hop_size = 512
    num_song_per_genre = 1900
    ext_type = 'mp3'
    preprocUnlabeledDataset(UNLABELED_DATA_DIR, sr, window_size, hop_size, num_song_per_genre, ext_type)
    print('Finished')
    return()

if __name__ == "__main__":
    print('main is running directly')
    main()

