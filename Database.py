import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pickle
from sklearn.externals import joblib


class Database(object):
    def __init__(self):
        # Udacity dataset
        self.pathToDataset = "../data/object-dataset/"
        self.images = glob.glob(self.pathToDataset + "*.jpg")
        assert(len(self.images) != 0)
        # Udacity project dataset
        self.pathToVehicles = "../data/vehicles_smallset/"#cars1/"
        self.pathToNonVehicles = "../data/non-vehicles_smallset/"#notcars1/"
        self.cars = glob.glob(self.pathToVehicles + "**/*.jpeg", recursive=True)
        self.notcars = glob.glob(self.pathToNonVehicles + "**/*.jpeg", recursive=True)
        assert(len(self.cars) != 0)
        assert(len(self.notcars) != 0)
        # Small project test images - Nb: 6
        self.pathToTestImages = "./test_images/"
        self.testImages = glob.glob(self.pathToTestImages + "*.jpg")
        assert(len(self.testImages) != 0)
        # Path to input video
        #self.inputVideo = "./project_video.mp4"
        self.inputVideo = "./test_video.mp4"
        # Path to output video
        #self.outputVideoName = "./output_videos/project_video.mp4"
        self.outputVideoName = "./output_videos/test_video.mp4"
        # Image data
        self.imageSize = self.GetImageSize()
        # Classifier and scaler pickle saved
        self.classifierPickleName = './saved_data/classifier.p'
        self.scalerPickleName = './saved_data/scaler.p'


    def GetOutputVideoPath(self):
        return self.outputVideoName

    def GetImageSize(self):
        '''
        Probably differentiate between training and
        test images...
        '''
        return self.GetRandomImage().shape

    def GetRandomImage(self, path=None):
        if path is None:
            images = self.testImages
        else:
            images = glob.glob(path)
        image = np.random.randint(0, len(images))
        return mpimg.imread(images[image])

    def GetListOfImages(self):
        '''
        Returns 2 lists:
        * file names of images with cars
        * file names of images without cars
        '''
        cars = []
        notcars = []
        cars = self.cars
        notcars = self.notcars
        if 0: # Maybe for later
            for image in self.images:
                if 'image' in image or 'extra' in image:
                    notcars.append(image)
                else:
                    cars.append(image)

        assert(len(cars) > 0)
        assert(len(notcars) > 0)
        return cars, notcars

    def SaveObject(self, obj, path):
        # save the classifier
        # pickle.dump(obj, open(self.classifierPickleName, 'wb'))
        joblib.dump(obj, path)
        print("Object saved to pickle file : " + str(path))
        # with open(self.classifierPickleName, 'wb') as fid:
        #     cPickle.dump(obj, fid)

    def LoadObject(self, path):
        # load it again
        print("Object loaded from pickle file : " + str(path))
        # return pickle.load(open(self.classifierPickleName, 'rb'))
        return joblib.load(path)
        # with open(self.classifierPickleName, 'rb') as fid:
        #     return cPickle.load(fid)
