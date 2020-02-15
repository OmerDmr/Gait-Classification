import sklearn
from sklearn.model_selection import train_test_split
from keras import Sequential, optimizers, utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Activation, Flatten, Dense, Conv2D, Lambda, Cropping2D, Dropout, LeakyReLU, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle, randint
import numpy as np
import cv2
import csv
import os
from os import walk
from os import path
import time
from sklearn.preprocessing import StandardScaler
from skimage.filters import threshold_otsu
from skimage import measure
from keras.models import model_from_json
from unet import *

def CNN(inputShape):

	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same',input_shape=(64,64,3)))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(103))
	model.add(Activation('softmax'))

	model.summary()

	return model

def getImages(filePath):
	persons = []
	classes = []
	i = -1
	for dirs in sorted(os.listdir(filePath)):
		clsId = []
		path1 = filePath + dirs
		i = i+1
		for dirs2 in sorted(os.listdir(path1)):
			path2 = path1 +'/'+ dirs2
			for root, dirs, files in os.walk(path2):
				feature = []
				for name in sorted(files):
					if name.endswith((".jpg",".jpeg",".png",".tiff")):
						baseName=os.path.join(root,name)
						print(baseName)
						img = cv2.imread(baseName)
						img = cv2.resize(img,(64,64))
						feature.append(img)
						classes.append(i)
				print(np.shape(feature))
				clsId.append(feature)
		print(np.shape(clsId))
		persons.append(clsId)
	return persons,classes


def trainAndTest():

	filePath = "/home/omer/gaitRec/persons/"
	X,y = getImages(filePath)
	print(np.asarray(X).shape,np.asarray(y).shape)
	
              

	sample,a1,a2,a3,a4,a5=np.shape(X)
	X = np.array(X).reshape(sample*a1*a2,a3,a4,a5)
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=rand_state)

	datagen = ImageDataGenerator()

	datagen.fit(X_train)

	valgen = ImageDataGenerator()
	valgen.fit(X_test)
	
	model = unet(input_size = (64,64,3))
	
	
	batch_size = 32
	inputShape = (64,64)
	#model = CNN(inputShape)

	checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_acc',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')

	model.compile(loss='sparse_categorical_crossentropy',
				optimizer=Adam(lr=1.0e-4), metrics=['accuracy'])
	

	model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
						steps_per_epoch=len(X_train)/batch_size, validation_data=valgen.flow(X_test, y_test, batch_size=batch_size),
						validation_steps=len(X_test)/batch_size, epochs=30, callbacks=[checkpoint])

	
	model.compile(loss='sparse_categorical_crossentropy',
				optimizer='adam', metrics=['accuracy'])

	model.fit(X,y,epochs=30,batch_size=batch_size,verbose=0)
	scores = model.evulate(X,y,verbose = 0)
	print("%s: %2f%%"%(model.metrics_names[1],score[1]*100))
	
	model_json = model.to_json()
	with open("model.json","w") as json_file:
		json_file.write(model_json)
	model.save_weights("model.h5")
	

trainAndTest()








