from scipy.ndimage.measurements import label
import numpy as np
import cv2

class Filter(object):
    """docstring for Filter."""
    def __init__(self, threshold=1):
        super(Filter, self).__init__()
        self.threshold = threshold
        self.heatMap = None
        self.labels = None
        self.fadeOutFactor = 0.65
        self.newBBoxWeight = 1
        self.Summary()

    def Summary(self):
        '''
        Prints some of the parameters
        '''
        print("######################")
        print("##      Filter      ##")
        print("# ")
        print("# Filter threshold : " + str(self.threshold))
        print("# Fade out factor between frames : " + str(self.fadeOutFactor))
        print("# New bounding box weight : " + str(self.newBBoxWeight))
        print("######################")
        print()

    def AddHeat(self, bboxes, weight=None):
        if weight is None:
            weight = self.newBBoxWeight

        # Iterate through list of bboxes
        for box in bboxes:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heatMap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += weight
        # Crop the heap map to keep it with observable range
        self.heatMap = np.clip(self.heatMap, 0, 255)


    def ApplyThreshold(self):
        # Zero out pixels below the threshold
        self.heatMap[self.heatMap <= self.threshold] = 0


    def LabelHeatMap(self):
        '''
        labels is a 2-tuple, where the first item is
        an array the size of the heatmap input image
        and the second element is the number of labels
        (cars) found.
        '''
        self.labels = label(self.heatMap)

# TODO: Move to renderer ?
    def DrawLabeledBBoxes(self, img):
        # Iterate through all detected cars
        for car_number in range(1, self.labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (self.labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img


    def FilterBBoxes(self, image, bboxes):
        if self.heatMap is None:
            self.heatMap = np.zeros_like(image[:,:,0]).astype(np.float)
            self.AddHeat(bboxes, 1)
        else:
            self.heatMap = self.heatMap * self.fadeOutFactor
            self.AddHeat(bboxes)
        self.ApplyThreshold()
        self.LabelHeatMap()
        return self.DrawLabeledBBoxes(np.copy(image)), self.heatMap

    def BBoxesAndHeatMap(self, image, bboxes):
        self.heatMap = np.zeros_like(image[:,:,0]).astype(np.float)
        self.AddHeat(bboxes, 1)
        return self.heatMap
