from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import Counter
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score



digits = pd.read_csv("train.csv")

features = np.array(digits.ix[:,1:], 'int16') 
labels = np.array(digits['label'], 'int')

features = np.multiply(features, 1.0 / 255.0)

print features.shape
# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(5, 5), cells_per_block=(2,2), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print "Before clf"

clf = MLPClassifier(hidden_layer_sizes=(1000), random_state=1,early_stopping=True)

#clf.fit(features,labels)

scores = cross_val_score(clf, hog_features, labels, cv=5)
print scores
print scores.mean()
