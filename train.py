import os 
import pickle
from pydub import AudioSegment
import scipy.io.wavfile
from python_speech_features import mfcc
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
def train_model():
	directories = ['classical', 'metal', 'country']
	classical_wav_files = [directories[0] + '/' + file for file in os.listdir(directories[0]) if file.endswith('wav')]
	classical_train_set, classical_test_set = classical_wav_files[:70], classical_wav_files[70:]


	metal_wav_files = [directories[1] + '/' + file for file in os.listdir(directories[1]) if file.endswith('wav')]
	metal_train_set, metal_test_set = metal_wav_files[:70], metal_wav_files[70:]

	country_wav_files = [directories[2] + '/' + file for file in os.listdir(directories[2]) if file.endswith('wav')]
	country_train_set, country_test_set = country_wav_files[:70], country_wav_files[70:]

	train_set = classical_train_set + metal_train_set + country_train_set
	test_set = classical_test_set + metal_test_set + country_test_set
	X_list = []
	Y_list = []
	for file in train_set:
	    sample_rate, X = scipy.io.wavfile.read(file)
	    mfcc_features = mfcc(X, sample_rate)
	    X_list.append(mfcc_features.flatten()[:30000])
	    if 'classical' in file:
	        Y_list.append(0)
	    elif 'metal' in file:
	        Y_list.append(1)
	    else:
        	Y_list.append(2)

	X = np.stack(X_list)
	Y = np.array(Y_list)
	lda = LinearDiscriminantAnalysis()
	lda.fit(X, Y)
	filename = 'finalized_model_lda.sav'
	pickle.dump(lda, open(filename, 'wb'))
	print('Model Successfuly Saved')

train_model()