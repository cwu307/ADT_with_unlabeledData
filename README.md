# ADT_with_unlabeledData
CW @ GTCMT 2018
Latest update: July 2018


## Introduction 

ADT_with_unlabeledData includes Python implementations for experimenting the viability of integrating unlabeled data to Automatic Drum Transcription (ADT) systems. This includes two major methods: student-teacher learning and feature learning. For related information, please refer to the following publication:

Chih-Wei Wu and Alexander Lerch, From labeled to unlabeled data -- On the Data Challanege in Automaatic Drum Transcription, in Proceedings of the International Society of Music Information Retrieval Conference (ISMIR), Paris, 2018 (accpeted, to be presented)

## Instruction 

This repository includes all the pre-trained machine learning models for conducting the experiments. However, due to space limitation, the preprocessed data and the feature matrices for reproducing the experiments are not included in this repository, and the training processes cannot be repeated without these datasets. To gain access to these datasets, please contact the author (contact information below). 

In addition to the models, this repository also includes the intermediate results from all the evaluated systems. To repeat the evaluation process, please enter the following commands in the terminal: 

    >> cd evaluation
    >> python runEvaluation.py -f 'studentTeacher' 
or 
    >> python runEvaluation.py -f 'featureLearning'

These commands will compare the groundtruth of the labeled datasets with the prediction results of the evaluated systems and generate the reports in
'./evaluation/evaluationResults/'

These results can also be found in './evaluation/archivedEvalResults/'

The following sections describe the content of each subfolder in this repository:

### './evaluation/':
This folder includes the main scripts for evaluation. The main script compares the output from each evaluated system with the corresponding groundtruth for each dataset, and subsequently generates a .txt file to summarize the evaluation results.   

### './preprocessData/': 
1) getPreprocData.py: this script preprocess the unlabeled dataset and save the STFT for later use. To run this script, the entire unlabeled dataset is required.
2) getDataSplits.py: this script splits the dataset into training and testing sets; the list of these splits are saved as .npy files for later use
3) FileUtil.py: a script that includes some utility functions

### './featureLearning/': 
This folder is for the "feature learning" paradigm and its related experiments:
1) './featureLearning/autoencoder/': the scripts in this folder are used for training a convolutional autoencoder for extracting features
2) './featureLearning/featureExtraction/': the scripts in this folder use the pre-trained autoencoder to extract features from the labeled dataset. Note that the feature matrices are not included here due to the space constraint
3) './featureLearning/mainTaskModels/': using the extracted feature matrices, multiple SVM classifiers are trained. The pre-trained models are saved in './trainedClassifier/' subfolder.
4) runPrediction.py: this script uses the pre-trained classifiers to predict the "held-out" dataset. The results are stored in './featureLearning/predictionResults/' 

### './studentTeacher/':
This folder is for the "student teacher learning" paradigm and its related experiments:
1) './studentTeacher/teacher/': runTeacher.py uses the teacher models to predict the unlabeled data and generates soft-targets. The soft-targets are not included here due to the space constraints
2) './studentTeacher/student/': trainStudent.py uses the soft-targets and the unlabeled dataset to train a fully connected deep neural network. The trained models are stored in './savedStudentModels'
3) runPredictionpy: this script uses the pre-trained student models to predict the test dataset. The results are stored in './studentTeacher/predictionResults/'

### Development Environment:
- Ubuntu 16.04.4
- Python 2.7.12

## Dependencies
see requirement.txt

## Contact

Chih-Wei Wu
Georgia Tech Center for Music Technology (GTCMT)
cwu307@gatech.edu

840 McMillan Street
Atlanta, GA, USA, 30332


