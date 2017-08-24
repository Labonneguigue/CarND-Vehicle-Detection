import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import Classifier as C
import Database as D
import Renderer as R
import Utils as U
import VehicleDetector as V

class Test(object):
    """docstring for Test."""
    def __init__(self):
        super(Test, self).__init__()
        self.database = D.Database()
        cars, notcars = self.database.GetListOfImages()
        self.classifier = C.Classifier(cars, notcars, loadFromFile=True, database=self.database)
        self.renderer = R.Renderer()
        self.vehicleDetector = V.VehicleDetector()

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
        cars_features = U.Utils.ExtractListImagesFeatures(cars)
        notcars_features = U.Utils.ExtractListImagesFeatures(notcars)
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
        self.database.SaveObject(self.classifier.classifier, self.database.classifierPickleName)
        print("End of Train classifier test.")
        print()

    def TestSlidingWindows(self, img):
        print("Run Sliding Windows test:")
        # bboxes128 = U.Utils.GetSlidingWindows(img,
        #                     y_start_stop=(img.shape[0]*2/5, img.shape[0]),
        #                     xy_window=(128, 128),
        #                     xy_overlap=(0.75, 0.75))
        # bboxes64 = U.Utils.GetSlidingWindows(img,
        #                     y_start_stop=(img.shape[0]*2/5, img.shape[0]),
        #                     xy_window=(64, 64),
        #                     xy_overlap=(0.75, 0.75))
        bboxes = self.vehicleDetector.LoadSlidingWindows(verbose=True)
        result = R.Renderer.DrawSlidingBoxes(img, bboxes, random=True)
        self.renderer.SaveImage(result, "sliding_window-sparse.png")
        result = R.Renderer.DrawSlidingBoxes(img, bboxes)
        self.renderer.SaveImage(result, "sliding_window.png")
        # result = R.Renderer.DrawSlidingBoxes(img, bboxes128, color=(255, 0, 0), random=True)
        # self.renderer.SaveImage(result, "sliding_window-128.png")
        print("End of Sliding windows test.")
        print()


    def TestClassification(self):
        print("Run Classification test:")
        #vehicleDetector = V.VehicleDetector()
        for img in self.database.testImages:
            result = self.vehicleDetector.ProcessImage(mpimg.imread(img), filtering=False)
            self.renderer.SaveImage(result, os.path.basename(img))
        print("End Classification test.")
        print()


    def TestHeatMap(self):
        print("Run Heat Map test:")
        # vehicleDetector = V.VehicleDetector()
        for img in self.database.testImages:
            resultBBoxes, heatMap = self.vehicleDetector.ProcessImage(mpimg.imread(img), agg=False)
            self.renderer.SaveImagesSideBySide(resultBBoxes, heatMap,
                                os.path.basename(img),
                                "Thresholded bounding boxes",
                                "Heat Map")
        print("End Heat Map test.")
        print()

    def TestProcessImagePipeline(self):
        print("Run Process Image Pipeline test:")
        # vehicleDetector = V.VehicleDetector()
        for img in self.database.testImages:
            result = self.vehicleDetector.ProcessImage(mpimg.imread(img))
            self.renderer.SaveImage(result,
                                os.path.basename(img))
        print("End Process Image Pipeline test.")
        print()

    def RunTests(self):
        print("")
        print("****************************")
        print("********** TESTS ***********")
        print("****************************")
        print("")

        if 0:
            self.TestHOG(self.database.GetRandomImage(self.database.cars))
        if 0:
            self.TestNormalizationFeatures()
        if 1:
            self.TestTrainClassifier()
        if 0:
            self.TestSlidingWindows(self.database.GetRandomImage())
        if 0:
            self.TestClassification()
        if 0:
            self.TestHeatMap()
        if 0:
            self.TestProcessImagePipeline()

        print()
        print("****************************")
        print("******* END OF TESTS *******")
        print("****************************")


if __name__ == "__main__":
    test = Test()
    test.RunTests()
