import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

class Utils(object):

    def __init__(self):
        pass

    @staticmethod
    def BinSpatial(img, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        #print("Spatial bin - Image dim : " + str(img.shape) + " output size : " + str(size))
        assert(len(size) == 2)
        features = cv2.resize(img, size).ravel()
        # Return the feature vector
        return features


    @staticmethod
    def ColorHist(img, nbins=32, bins_range=(0, 256), richOutput=False):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0],
                                        channel2_hist[0],
                                        channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        if richOutput:
            # Generating bin centers
            bin_edges = channel1_hist[1]
            bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
            return hist_features, channel1_hist, channel1_hist, channel1_hist, bin_centers
        else:
            return hist_features

    @staticmethod
    def GetHOGFeatures(img, nbOrientation, pix_per_cell, cell_per_block, ravel=True, visualise=False):
        '''
        feature_vector set to true would .ravel() the output
        so that the output vector is 1D
        '''
        assert(len(img.shape)==2)
        if visualise:
            features, hog_image = hog(img,
                    orientations=nbOrientation,
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block),
                    transform_sqrt=False,
                    visualise=visualise,
                    feature_vector=ravel)
            return features, hog_image
        else:
            features = hog(img,
                    orientations=nbOrientation,
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block),
                    transform_sqrt=False,
                    visualise=visualise,
                    feature_vector=ravel)
            return features

    @staticmethod
    def ExtractListImagesFeatures(imgs, cspace='YUV', spatial_size=(32, 32),
                            hist_bins=32, hist_range=(0, 256),
                            hog_channel='ALL', nbOrientation=9,
                            pix_per_cell=8, cell_per_block=2):
        '''
        Define a function to extract features from a list of images
        Have this function call bin_spatial() and color_hist()
        '''
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = mpimg.imread(file)
            # Append the new feature vector to the features list
            features.append(Utils.ExtractFeatures(image, cspace, spatial_size,
                                    hist_bins, hist_range,
                                    hog_channel, nbOrientation,
                                    pix_per_cell, cell_per_block))
            #features.append(hog_features)
        # Return list of feature vectors
        return features

    @staticmethod
    def ExtractFeatures(image, cspace='YUV', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256),
                        hog_channel='ALL', nbOrientation=9,
                        pix_per_cell=8, cell_per_block=2):
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(image)
        # Apply bin_spatial() to get spatial color features
        spatial_features = Utils.BinSpatial(feature_image)
        # Apply color_hist() also with a color space option now
        hist_features = Utils.ColorHist(feature_image,
                                       nbins=hist_bins,
                                       bins_range=hist_range)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(Utils.GetHOGFeatures(feature_image[:,:,channel],
                                    nbOrientation, pix_per_cell, cell_per_block,
                                    visualise=False, ravel=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = Utils.GetHOGFeatures(feature_image[:,:,hog_channel],
                                    nbOrientation, pix_per_cell,
                                    cell_per_block, visualise=False, ravel=True)


        # Return the feature list
        return np.concatenate((spatial_features, hist_features, hog_features))
        #return hog_features

    @staticmethod
    def NormalizeFeatures(car_features, notcar_features, scaler):
        '''
        Normalize each set of features so that they have equal variance
        and 0 mean.
        '''
        assert(len(car_features)>0)
        assert(len(notcar_features)>0)
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        scaler.fit(X)
        # Apply the scaler to X
        return scaler.transform(X), X


    @staticmethod
    def GetSlidingWindows(img, x_start_stop=[None, None], y_start_stop=[None, None],
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        '''
        This function returns a list of bounding boxes that can be used
        to obtain small patchs of the image and use them to determine
        if they contain a car or not

        Parameters:
        x_start_stop and y_start_stop determine the region of the image
            that is considered when creating the bounding boxes
        xy_window is the size of the bounding boxes
        xy_overlap is the overlapping between 2 consecutive bounding boxes.
        '''
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = int(xs*nx_pix_per_step + x_start_stop[0])
                endx = int(startx + xy_window[0])
                starty = int(ys*ny_pix_per_step + y_start_stop[0])
                endy = int(starty + xy_window[1])
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    # @staticmethod
    # def LoadCarNonCarFeatures():
    #
