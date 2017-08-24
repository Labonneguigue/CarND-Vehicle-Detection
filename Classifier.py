import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
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
    def __init__(self, cars, notcars, loadFromFile=False, database=None):
        super(Classifier, self).__init__()
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.nbOrientation = 9
        self.testOverTrainingSetRatio = 0.1
        self.cSpace = 'YUV'
        self.histBins = 32
        self.spatialSize = (32, 32)
        self.hogChannels = 'ALL'
        # Scaler
        self.scaler = None
        #classifier
        self.classifierType = 'SVC' # SVC or LinearSVC
        self.classifier = self.CreateClassifier(cars, notcars, loadFromFile, database)


    def CreateClassifier(self, cars, notcars, loadFromFile=False, database=None):
        if self.classifierType == 'SVC':
            if loadFromFile:
                self.classifier = database.LoadObject(database.classifierPickleName)
                self.scaler = database.LoadObject(database.scalerPickleName)
            else:
                self.classifier = SVC()
                self.TrainClassifier(cars, notcars)
                if (database is not None):
                    database.SaveObject(self.classifier, database.classifierPickleName)
                    database.SaveObject(self.scaler, database.scalerPickleName)

            return self.classifier

        else:
            print("Not implemented")

    def TrainClassifier(self, cars, notcars, classifier='SVC'):
        '''
        @param  cars        list of image names containing a car
        @param  notcars     list of image names not containing a car
        @param  classifier  classifier to use
        '''
        print("Extracting features ...")
        cars_features = U.Utils.ExtractListImagesFeatures(cars,
                                                cspace=self.cSpace,
                                                spatial_size=self.spatialSize,
                                                hist_bins=self.histBins,
                                                hog_channel=self.hogChannels,
                                                nbOrientation=self.nbOrientation,
                                                pix_per_cell=self.pix_per_cell,
                                                cell_per_block=self.cell_per_block)
        notcars_features = U.Utils.ExtractListImagesFeatures(notcars,
                                                    cspace=self.cSpace,
                                                    spatial_size=self.spatialSize,
                                                    hist_bins=self.histBins,
                                                    hog_channel=self.hogChannels,
                                                    nbOrientation=self.nbOrientation,
                                                    pix_per_cell=self.pix_per_cell,
                                                    cell_per_block=self.cell_per_block)
        print("Normalizing features ...")
        self.scaler = StandardScaler()
        scaled_X, X = U.Utils.NormalizeFeatures(cars_features, notcars_features, self.scaler)
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

        # parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        #parameters = {'C':[0.8,0.9, 1, 1.1]}
        #svr = svm.SVC()
        #self.classifier = GridSearchCV(svr, parameters)
        #self.classifier = svm.SVC(C=0.9)
        self.classifier = LinearSVC(C=0.9)
        self.classifier.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.classifier.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        # print("Best params : ")
        # print(self.classifier.best_params_)
        t=time.time()
        n_predict = 10
        print('My SVC predicts:    ', self.classifier.predict(X_test[0:n_predict]))
        print('For these',n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
        print()

    def SaveClassifierToFile(self, database):
        '''
        Calls the database to store the classifier to avoid
        retraining it each time
        '''
        database.SaveClassifierToPickle(self.classifier)
