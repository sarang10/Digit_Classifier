from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

digits = pd.read_csv("train.csv")

features = np.array(digits.ix[:,1:], 'int16') 
labels = np.array(digits['label'], 'int')

features = np.multiply(features, 1.0 / 255.0)

list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14,14), cells_per_block=(8,8), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

print hog_features.shape

clf = LinearSVC()

clf.fit(hog_features, labels)

digits = pd.read_csv("test.csv")

features = np.array(digits.ix[:,0:], 'int16') 

features = np.multiply(features, 1.0 / 255.0)

list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(8,8), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

results = clf.predict(hog_features)

print hog_features.shape

np.savetxt('sub_try.csv', np.c_[range(1,len(hog_features)+1),results], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

