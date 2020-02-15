import cv2
import numpy as np
import os
from os import walk,path
import glob
from skimage.feature import hog


def getImages(filePath,classes):
	persons = []
	i=-1
	for dirs in sorted(os.listdir(filePath)):
		i=i+1
		clsId = []
		path1 = filePath + dirs
		for dirs2 in sorted(os.listdir(path1)):
			path2 = path1 +'/'+ dirs2
			for root, dirs, files in os.walk(path2):
				feature = []
				for name in sorted(files):
					if name.endswith((".jpg",".jpeg",".png",".tiff")):
						baseName=os.path.join(root,name)
						print(baseName)
						img = cv2.imread(baseName,0)
						img = cv2.resize(img,(64,64))
						ftr = getHogFeatures(img)
						feature.append(ftr)
						classes.append(i)
				print(np.shape(feature))
				clsId.append(feature)
		print(np.shape(clsId))
		persons.append(clsId)
	return persons,classes
		
def getHogFeatures(img):

	features = hog(img, orientations=9, 
					pixels_per_cell=(8,8),
					cells_per_block=(2,2), 
					transform_sqrt=True, 
					visualise=False, feature_vector=True)

	return features



