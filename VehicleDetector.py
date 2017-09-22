import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

import Classifier as C
import Database as D
import Renderer as R
import Utils as U
import Filter as F

class VehicleDetector(object):
    """docstring for VehicleDetector."""
    def __init__(self):
        super(VehicleDetector, self).__init__()
        # Sub-components
        self.renderer = R.Renderer()
        self.database = D.Database()
        self.filter = F.Filter()
        cars, notcars = self.database.GetListOfImages()
        self.classifier = C.Classifier(cars, notcars, loadFromFile=True, database=self.database)
        # Output video parameters
        self.outputToImages = 0
        self.outputVideoName = self.database.GetOutputVideoPath()
        # Sliding windows
        self.yStart = 400
        self.yStop = 650
        self.x_overlap = 0.5
        self.y_overlap = 0.75
        self.bboxes = self.LoadSlidingWindows()
        # Test images

        # Train classifier ?
        self.trainClassifier = 1
        # TODO: implement the loading

    def LoadSlidingWindows(self, verbose=False):
        img = self.database.GetRandomImage()
        bboxes = []
        bboxes = U.Utils.GetSlidingWindows(img,
                            y_start_stop=(self.yStart, self.yStop),
                            xy_window=(128, 128),
                            xy_overlap=(self.x_overlap, self.y_overlap))
        if verbose:
            print(str(len(bboxes)) + " 128 bboxes.")

        bboxes64 = U.Utils.GetSlidingWindows(img,
                            y_start_stop=(self.yStart, self.yStart+120),
                            xy_window=(64, 64),
                            xy_overlap=(self.x_overlap, self.y_overlap))
        bboxes.extend(bboxes64)

        if verbose:
            print(str(len(bboxes64)) + " 64 bboxes.")

        bboxes192 = U.Utils.GetSlidingWindows(img,
                            y_start_stop=(self.yStart, self.yStop),
                            xy_window=(192, 192),
                            xy_overlap=(self.x_overlap, self.y_overlap))
        if verbose:
            print(str(len(bboxes192)) + " 192 bboxes.")

        bboxes.extend(bboxes192)
        # bboxes = np.hstack(bboxes128,bboxes64)
        assert(len(bboxes) != 0)
        print("The VehicleDetector uses " + str(len(bboxes)) + " sliding windows.")
        return bboxes

    def DetectCars(self, img):
        '''
        Run the classifier in predict mode on each sub-images
        in the provided image
        '''
        assert(self.classifier.scaler != None)
        assert(self.classifier.classifier != None)
        detections = []
        #i = 0
        for window in self.bboxes:
            # print(window)
            # print(img.shape)
            #print(window)
            currentlyTestedImg = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            imgFeatures = U.Utils.ExtractFeatures(currentlyTestedImg,
                                spatial_size=(self.classifier.spatialSize, self.classifier.spatialSize),
                                hist_bins=self.classifier.histBins,
                                nbOrientation=self.classifier.nbOrientation,
                                pix_per_cell=self.classifier.pix_per_cell,
                                cell_per_block=self.classifier.cell_per_block)
            testedFeatures = self.classifier.scaler.transform(np.array(imgFeatures).reshape(1, -1))
            prediction = self.classifier.classifier.predict(testedFeatures)
            #self.renderer.SaveImage(currentlyTestedImg, "../temp/" + str(i) + ".jpeg")
            if prediction == 1:
                detections.append(window)
            #i+=1
        return detections

    def ProcessImage(self, image, key_frame_interval=20, cache_length=10, filtering=True, agg=False, heatmap=False):
        '''
        Entire processing pipeline
        '''
        hot_windows = self.DetectCars(image)
        if filtering:
            resultBBoxes, heatMap = self.filter.FilterBBoxes(image, hot_windows)
            if agg:
                result = self.renderer.AggregateViews([resultBBoxes, heatMap])
                return result
            elif heatmap:
                return resultBBoxes, heatMap
            else:
                return resultBBoxes

        else:
            return R.Renderer.DrawSlidingBoxes(image, hot_windows)

    def OutputImages(self, image, key_frame_interval=20, cache_length=10):
        '''
        Entire processing pipeline and output each output
        images to an image file
        '''
        pass
        print("Not implemented.")

    def ProcessVideo(self):
        print('Processing video ... ' + self.database.inputVideo)
        vfc = VideoFileClip(self.database.inputVideo)#.subclip(27, 32)
        if self.outputToImages:
            detected_vid_clip = vfc.fl_image(self.OutputImages)
        else:
            detected_vid_clip = vfc.fl_image(self.ProcessImage)
        detected_vid_clip.write_videofile(self.outputVideoName, audio=False)


if __name__ == "__main__":
    vehicleDetector = VehicleDetector()
    vehicleDetector.ProcessVideo()
