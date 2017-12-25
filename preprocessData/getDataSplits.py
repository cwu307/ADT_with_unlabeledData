'''
this script takes the preprocessed data (stft) in .npy format
randomly shuffles the entire dataset, and returns three lists:
1) training
2) validation
3) test
CW @ GTCMT 2017
'''
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
GENRES = ['alternative-songs', 'dance-club-play-songs', 'hot-mainstream-rock-tracks', 'latin-songs', 'pop-songs', 'r-b-hip-hop-songs']


def concatenateMatrix(matrix_con, matrix, selected_axis):
    if len(matrix_con) == 0:
        matrix_con = matrix
    else:
        matrix_con = np.concatenate((matrix_con, matrix), axis=selected_axis)
    return matrix_con

'''
different input representations should have the same file path with different folder names
this function simply swap the original folder name with the new one
all_lists_mod = modifyLists(all_lists, original_folder, target_folder)
input:
    all_lists: original list from the getDataSplits()
    original_folder: str, name of the original stft folder (e.g., /stft/)
    target_folder: str,  name of the target folder (e.g., /stft_p/) 
output
    all_lists_mod: a modified all_lists 
'''
def modifyLists(all_lists, original_folder, target_folder):
    all_lists_mod = []
    print('==== swapping %s for %s in given lists' % (original_folder, target_folder))
    for one_list in all_lists:
        new_list = [(filepath.replace(original_folder , target_folder), label) for filepath, label in one_list]
        all_lists_mod.append(new_list)
    return all_lists_mod


'''
this function checks whether the modified list is still consistent with the original one
bool = isCompatible(all_lists,  all_lists_mod)
input:
    all_lists:  original train-test splits
    all_lists_mod: modified train-test splits (only modify the folder names)
output:
    bool: true/false 
'''
def isCompatible(all_lists,  all_lists_mod):
    problem_count  = 0
    for i in range(0, len(all_lists)):
        original_list = all_lists[i]
        mod_list = all_lists_mod[i]
        if len(original_list) != len(mod_list):
            problem_count += 1
        for j in range(0, len(original_list)):
            tmp, label = original_list[j]
            filename = tmp.split('/')[-1]
            tmp2, label2 = mod_list[j]
            filename2 = tmp2.split('/')[-1]
            if filename != filename2:
                problem_count += 1
            if label != label2:
                problem_count += 1
    if problem_count == 0:
        return True
    else:
        return False

def checkGenreBalance(all_lists):
    dist = {}
    for one_list in all_lists:
        for item, label in one_list:
            if label not in dist:
                dist[label] = 1
            else:
                dist[label] += 1
        print(dist)
        dist = {}
    return ()

'''
this function splits the entire dataset list into [training, validation, test] accordingly 
the resulting array is saved as .npy file
'''
def getDataSplits(DATA_DIR, num_songs_per_genre):
    all_lists = []
    all_training_set = []
    all_test_set = []
    all_validation_set = []
    print('==== going through all the genre folders')
    for genre in GENRES:
        print('- working on genre = %s ...' % genre)
        genre_path = DATA_DIR + genre + '/*.npy'
        all_files = glob(genre_path)
        np.random.shuffle(all_files) #this shuffling is probably redundant...
        all_files = all_files[0:num_songs_per_genre]
        all_files = [(path, genre) for path in all_files]
        print('- spliting data...')
        training_set, other = train_test_split(all_files, test_size=0.3, random_state=10)
        validation_set, test_set = train_test_split(other, test_size=0.5, random_state=10)
        print('- concatenating results...')
        all_training_set = concatenateMatrix(all_training_set, training_set, 0)
        all_test_set = concatenateMatrix(all_test_set, test_set, 0)
        all_validation_set = concatenateMatrix(all_validation_set, validation_set, 0)
    
    print('==== concatenation complete!')
    print('- there are %d training songs, %d validation songs, and %d test songs ...' % (len(all_training_set), len(all_validation_set), len(all_test_set)))
    print('==== now, shuffling all sets...')
    np.random.shuffle(all_training_set)
    np.random.shuffle(all_validation_set)
    np.random.shuffle(all_test_set)
    all_lists = [all_training_set, all_validation_set, all_test_set]
    print('==== done! returning all lists')
    return all_lists

def main():
    num_songs_per_genre = 1900
    DATA_DIR = './stft/'
    all_lists_stft =  getDataSplits(DATA_DIR, num_songs_per_genre)
    all_lists_stft_p = modifyLists(all_lists_stft, '/stft/', '/stft_p/')

    if isCompatible(all_lists_stft, all_lists_stft_p):
        print('- new lists is compatible with the original one') 
        print('==== saving results')
        np.save('stft_train_test_splits.npy',  all_lists_stft)
        np.save('stft_p_train_test_splits.npy', all_lists_stft_p)
    else:
        print('* something is wrong with the new list, results are not saved')
    return()


if __name__ == "__main__":
    print('main() is running directly')
    main()