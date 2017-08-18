import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import Utils as U

class Classifier(object):
    """
    docstring for Classifier.

    HOG:
    * Without block normalization
    Feature vector = nbOrientation * nbCells
    * With block normalization
    Feature vector = nbBlocks * nbCells/Block * nbOrientation
    """
    def __init__(self, cars, notcars):
        super(Classifier, self).__init__()
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.nbOrientation = 9
        self.testOverTrainingSetRatio = 0.1
        self.cSpace = 'RGB'
        self.histBins = 32
        self.spatialSize = (32, 32)
        self.hogChannels = 'ALL'
        #classifier
        self.classifierType = 'SVC' # Only choice so far
        self.classifier = self.CreateClassifier(cars, notcars)


    def CreateClassifier(self, cars, notcars):
        if self.classifierType == 'SVC':
            self.classifier = LinearSVC()
            self.TrainClassifier(cars, notcars)
            return self.classifier
        else:
            print("Not implemented")

    def TrainClassifier(self, cars, notcars, classifier='SVC'):
        '''
        @param  cars        list of image names containing a car
        @param  notcars     list of image names not containing a car
        @param  classifier  classifier to use
        '''
        cars_features = U.Utils.ExtractFeatures(cars,
                                                cspace=self.cSpace,
                                                spatial_size=self.spatialSize,
                                                hist_bins=self.histBins,
                                                hog_channel=self.hogChannels,
                                                nbOrientation=self.nbOrientation,
                                                pix_per_cell=self.pix_per_cell,
                                                cell_per_block=self.cell_per_block)
        notcars_features = U.Utils.ExtractFeatures(notcars,
                                                    cspace=self.cSpace,
                                                    spatial_size=self.spatialSize,
                                                    hist_bins=self.histBins,
                                                    hog_channel=self.hogChannels,
                                                    nbOrientation=self.nbOrientation,
                                                    pix_per_cell=self.pix_per_cell,
                                                    cell_per_block=self.cell_per_block)
        scaled_X, X = U.Utils.NormalizeFeatures(cars_features, notcars_features)

        # Define the labels vector
        y = np.hstack((np.ones(len(cars_features)), np.zeros(len(notcars_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
                                        test_size=self.testOverTrainingSetRatio,
                                        random_state=rand_state)

        print('Using spatial binning of:',self.spatialSize,
            'and', self.histBins,'histogram bins')
        print('Feature vector length:', len(X_train[0]))
        print('Training set length:', len(X_train))
        print('Test set length:', len(X_test))

        #if self.classifier = 'SVC':
        # Use a linear SVC
        # svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        self.classifier.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.classifier.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        n_predict = 10
        print('My SVC predicts:    ', self.classifier.predict(X_test[0:n_predict]))
        print('For these',n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

        # else:
        #     print()
        #     print("Not implemented")
        #     print()
