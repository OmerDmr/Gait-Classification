import cv2
import numpy as np
from getFeature import getImages
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle



filePath = "/home/omer/gaitRec/persons/"

#numberOfClasses = 103
#numberOfClassShape = 10
#numberOfAngleForEachClass = 11 [00,18,36,54,72,90,108,126,144,162,180]
#eachImage = 240x240x3

classes = []
features,labels = getImages(filePath,classes)
sample,a1,a2,a3= np.shape(features)


def trainAndTest(features,labels):
	X = features
	y = labels
	print(np.asarray(y).shape)
	X = np.array(X).reshape(sample*a1*a2,a3)
	print(np.asarray(X).shape)

	X, y = shuffle(X, y)
	
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

	svc = LinearSVC().fit(X_train,y_train)
	score1 = round(svc.score(X_test, y_test), 4) 
	print('Test Accuracy of SVC = ', score1)


trainAndTest(features,labels)








