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


class VehicleDetector(object):
    """docstring for VehicleDetector."""
    def __init__(self):
        super(VehicleDetector, self).__init__()
        # Sub-components
        self.renderer = R.Renderer()
        self.database = D.Database()
        cars, notcars = self.database.GetListOfImages()
        self.classifier = C.Classifier(cars, notcars)
        # Output video parameters
        self.outputToImages = 0
        self.outputVideoName = self.database.GetOutputVideoPath()
        # Sliding windows
        self.xy_overlap = 0.75
        self.bboxes = self.LoadSlidingWindows()
        # Test images

        # Train classifier ?
        self.trainClassifier = 1
        # TODO: implement the loading

    def LoadSlidingWindows(self):
        img = self.database.GetRandomImage()
        bboxes = []
        bboxes = U.Utils.GetSlidingWindows(img,
                            y_start_stop=(img.shape[0]*2/5, img.shape[0]),
                            xy_window=(128, 128),
                            xy_overlap=(self.xy_overlap, self.xy_overlap))
        bboxes.append(U.Utils.GetSlidingWindows(img,
                            y_start_stop=(img.shape[0]*2/5, img.shape[0]),
                            xy_window=(64, 64),
                            xy_overlap=(self.xy_overlap, self.xy_overlap)))
        assert(len(bboxes) != 0)
        print("The VehicleDetector uses " + str(len(bboxes)) + " sliding windows.")
        return bboxes

    def DetectCars(self, img):
        '''
        Run the classifier in predict mode on each sub-images
        in the provided image
        '''
        detections = []
        for window in self.bboxes:
            currentlyTestedImg = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            imgFeatures = U.Utils.ExtractFeatures(currentlyTestedImg,
                                spatial_size=(self.classifier.spatialSize, self.classifier.spatialSize),
                                hist_bins=self.classifier.histBins,
                                nbOrientation=self.classifier.nbOrientation,
                                pix_per_cell=self.classifier.pix_per_cell,
                                cell_per_block=self.classifier.cell_per_block)
            testedFeatures = scaler.transform(np.array(imgFeatures).reshape(1, -1))
            prediction = self.classifier.classifier.predict(testedFeatures)
            if prediction == 1:
                detections.append(window)
        return windows

    def ProcessImage(self, image, key_frame_interval=20, cache_length=10):
        '''
        Entire processing pipeline
        '''
        hot_windows = self.DetectCars(image)
        return U.Utils.DrawSlidingBoxes(hot_windows)

    def OutputImages(self, image, key_frame_interval=20, cache_length=10):
        '''
        Entire processing pipeline and output each output
        images to an image file
        '''
        pass
        print("Not implemented.")

    def ProcessVideo(self):
        print('Processing video ... ' + self.database.intputVideo)
        vfc = VideoFileClip(self.database.intputVideo).subclip(20, 24)
        if self.outputToImages:
            detected_vid_clip = vfc.fl_image(self.OutputImages)
        else:
            detected_vid_clip = vfc.fl_image(self.ProcessImage)
        detected_vid_clip.write_videofile(self.outputVideoName, audio=False)


if __name__ == "__main__":
    vehicleDetector = VehicleDetector()
    vehicleDetector.ProcessVideo()
