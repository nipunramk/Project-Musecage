import os
import pickle
from pydub import AudioSegment
import scipy.io.wavfile
import seaborn
import sklearn
import scipy
import librosa
import librosa.display
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from sklearn.metrics import confusion_matrix
from random import *

n_mfcc = 12


def train_model():
    # directories = ['blues', 'classical', 'country', 'disco',
    #                'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    directories = ['classical', 'country', 'disco', 'metal', 'pop']

    train_set = []
    test_set = []
    for i in range(len(directories)):
        wav_files = ['genres/' + directories[i] + '/' + file for file in os.listdir(
            'genres/' + directories[i]) if file.endswith('wav')]
        train_set.extend(wav_files[:70])
        test_set.extend(wav_files[70:])

    X, sample_rate = librosa.load(train_set[0], duration=30)
    mfcc_features = librosa.feature.mfcc(
        X, sr=sample_rate, n_mfcc=n_mfcc).T
    scaler = sklearn.preprocessing.StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc_features)

    ############################43% SCV

    # X_list = np.empty((0, n_mfcc), float)
    # Y_list = []
    #
    # for file in train_set:
    #     X, sample_rate = librosa.load(file, duration=30)
    #     mfcc_features = librosa.feature.mfcc(
    #         X, sr=sample_rate, n_mfcc=n_mfcc).T
    #     mfcc_scaled = scaler.transform(mfcc_features)
    #     X_list = np.vstack((X_list, mfcc_scaled))
    #     for i in range(len(directories)):
    #         if directories[i] in file:
    #             Y_list.extend([i for _ in mfcc_scaled])
    #
    # X = X_list
    # Y = np.array(Y_list)

    # SVC 36%
    # X_list = []
    # Y_list = []
    #
    # for file in train_set:
    #     X, sample_rate = librosa.load(file, duration=30)
    #     mfcc_features = librosa.feature.mfcc(
    #         X, sr=sample_rate, n_mfcc=n_mfcc).T
    #     mfcc_scaled = scaler.transform(mfcc_features).flatten()[:23000]
    #     X_list.append(mfcc_scaled)
    #     for i in range(len(directories)):
    #         if directories[i] in file:
    #             Y_list.append(i)
    #
    # X = np.stack([observation + np.random.normal(0, 0.04, 23000)
    #               for observation in X_list])
    # Y = np.array(Y_list)
    #
    # print(X.shape)
    # print(Y.shape)
    # svc = SVC(verbose=True)
    # svc.fit(X, Y)
    # # print(svc.score(X_list, Y_list))
    # filename = 'finalized_model_svc_5.sav'
    # pickle.dump(svc, open(filename, 'wb'))
    # filename = 'svc_scaler_5.sav'
    # pickle.dump(scaler, open(filename, 'wb'))
    # print('Model Successfuly Saved')

    svc = pickle.load(open('finalized_model_svc_5.sav', 'rb'))
    scaler = pickle.load(open('svc_scaler_5.sav', 'rb'))

    # SCV 43%
    # X_list = np.empty((0, n_mfcc), float)
    # Y_list = []
    #
    # for file in test_set:
    #     X, sample_rate = librosa.load(file, duration=30)
    #     mfcc_features = librosa.feature.mfcc(
    #         X, sr=sample_rate, n_mfcc=n_mfcc).T
    #     mfcc_scaled = scaler.transform(mfcc_features)[:100]
    #     X_list = np.vstack((X_list, mfcc_scaled))
    #     for i in range(len(directories)):
    #         if directories[i] in file:
    #             Y_list.extend([i for _ in mfcc_scaled])
    #
    # print(svc.score(X_list, Y_list))

    correct_test = 0
    total_test = 0

    for file in test_set:
        X, sample_rate = librosa.load(file, duration=30)
        mfcc_features = librosa.feature.mfcc(
            X, sr=sample_rate, n_mfcc=n_mfcc).T
        mfcc_scaled = scaler.transform(mfcc_features)
        predicted_labels = svc.predict(
            mfcc_scaled[int(len(mfcc_scaled) * .15):int(len(mfcc_scaled) * .85)])
        prediction = np.argmax([(predicted_labels == c).sum()
                                for c in range(len(directories))])
        for i in range(len(directories)):
            if directories[i] in file and i == prediction:
                correct_test += 1
        total_test += 1
        print(str(correct_test / total_test * 100) + '% correct')
    print(str(correct_test / total_test * 100) + '% correct')

    # SVC 36%
    # X_list = []
    # Y_list = []
    #
    # for file in test_set:
    #     X, sample_rate = librosa.load(file, duration=30)
    #     mfcc_features = librosa.feature.mfcc(
    #         X, sr=sample_rate, n_mfcc=n_mfcc).T
    #     mfcc_scaled = scaler.transform(mfcc_features).flatten()[:23000]
    #     X_list.append(mfcc_scaled)
    #     for i in range(len(directories)):
    #         if directories[i] in file:
    #             Y_list.append(i)
    #
    # X_list = np.stack(X_list)
    # Y_list = np.array(Y_list)
    #
    # print(svc.score(X_list, Y_list))


train_model()
