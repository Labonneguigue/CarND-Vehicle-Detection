import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import Classifier as C
import Database as D
import Renderer as R
import Utils as U


class Test(object):
    """docstring for Test."""
    def __init__(self):
        super(Test, self).__init__()
        self.classifier = C.Classifier()
        self.renderer = R.Renderer()
        self.database = D.Database()

    def TestHOG(self, image):
        '''
        Calls the HOG function of the classifier and saves the
        output by a call to the renderer
        '''
        print("Run HOG test:")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Call our function with vis=True to see an image output
        features, hog_image = self.classifier.GetHOGFeatures(gray,
                                                ravel=True,
                                                visualise=True)
        # ravel to False
        #features.shape = (149, 239, 2, 2, 9)
        #ravel to True
        # features.shape = (1281996,)
        self.renderer.SaveImagesSideBySide(image, hog_image, "hog_example.png",
                                            "Example Car Image",
                                            "HOG Visualization",
                                            grayscale=True)
        print("End of HOG test.")
        print()

    def TestNormalizationFeatures(self):
        '''
        Extracts features and plot them
        '''
        print("Run Features Normalization test:")
        cars, notcars = self.database.GetListOfImages()
        cars_features = U.Utils.ExtractFeatures(cars)
        notcars_features = U.Utils.ExtractFeatures(notcars)
        scaled_X, X = U.Utils.NormalizeFeatures(cars_features, notcars_features)
        car_ind = np.random.randint(0, len(cars))
        self.renderer.Save3ImagesSideBySide(mpimg.imread(cars[car_ind]),
                                            X[car_ind],
                                            scaled_X[car_ind],
                                            "normalized-features.png",
                                            titleB='Raw Features',
                                            titleC='Normalized Features'
                                            )
        print("End of Features Normalization test.")
        print()

    def TestTrainClassifier(self):
        print("Run Train Classifier test:")
        cars, notcars = self.database.GetListOfImages()
        self.classifier.TrainClassifier(cars, notcars)
        print("End of Train classifier test.")
        print()

    def TestSlidingWindows(self, img):
        print("Run Sliding Windows test:")
        bboxes = U.Utils.GetSlidingWindows(img)
        result = R.Renderer.DrawSlidingBoxes(img, bboxes)
        self.renderer.SaveImage(result, "sliding_window.png")
        print("End of Sliding windows test.")
        print()

    def RunTests(self):
        print("")
        print("****************************")
        print("********** TESTS ***********")
        print("****************************")
        print("")

        if 0:
            self.TestHOG(self.database.GetRandomImage())
        if 0:
            self.TestNormalizationFeatures()
        if 0:
            self.TestTrainClassifier()
        if 1:
            self.TestSlidingWindows(self.database.GetRandomImage())

        print()
        print("****************************")
        print("******* END OF TESTS *******")
        print("****************************")


if __name__ == "__main__":
    test = Test()
    test.RunTests()
