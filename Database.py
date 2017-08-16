import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
#from skimage.feature import hog
#from skimage import color, exposure
# images are divided up into vehicles and non-vehicles

# images = glob.glob('*.jpeg')
# cars = []
# notcars = []
#
# for image in images:
#     if 'image' in image or 'extra' in image:
#         notcars.append(image)
#     else:
#         cars.append(image)
#
#
# data_info = data_look(cars, notcars)
#
# print('Your function returned a count of',
#       data_info["n_cars"], ' cars and',
#       data_info["n_notcars"], ' non-cars')
# print('of size: ',data_info["image_shape"], ' and data type:',
#       data_info["data_type"])
# # Just for fun choose random car / not-car indices and plot example images
# car_ind = np.random.randint(0, len(cars))
# notcar_ind = np.random.randint(0, len(notcars))
#
# # Read in car / not-car images
# car_image = mpimg.imread(cars[car_ind])
# notcar_image = mpimg.imread(notcars[notcar_ind])
#
#
# # Plot the examples
# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(car_image)
# plt.title('Example Car Image')
# plt.subplot(122)
# plt.imshow(notcar_image)
# plt.title('Example Not-car Image')


class Database(object):
    def __init__(self):
        self.pathToDataset = "../data/object-dataset/"
        self.images = glob.glob(self.pathToDataset + "*.jpg")
        self.pathToVehicles = "../data/vehicles_smallset/cars1/"
        self.pathToNonVehicles = "../data/non-vehicles_smallset/notcars1/"
        self.cars = glob.glob(self.pathToVehicles + "*.jpeg")
        self.notcars = glob.glob(self.pathToNonVehicles + "*.jpeg")

    def data_look(self, car_list, notcar_list):
        '''
        Define a function to return some characteristics of the dataset
        '''
        data_dict = {}
        # Define a key in data_dict "n_cars" and store the number of car images
        data_dict["n_cars"] = len(car_list)
        # Define a key "n_notcars" and store the number of notcar images
        data_dict["n_notcars"] = len(notcar_list)
        # Read in a test image, either car or notcar
        example_img = mpimg.imread(car_list[0])
        # Define a key "image_shape" and store the test image shape 3-tuple
        data_dict["image_shape"] = example_img.shape
        # Define a key "data_type" and store the data type of the test image.
        data_dict["data_type"] = example_img.dtype
        # Return data_dict
        return data_dict

    def GetRandomImage(self):
        image = np.random.randint(0, len(self.images))
        return mpimg.imread(self.images[image])

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
