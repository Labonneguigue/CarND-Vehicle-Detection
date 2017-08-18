import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

class Renderer(object):
    """docstring for Renderer."""
    def __init__(self):
        super(Renderer, self).__init__()
        self.outputFolder = "./output_images/"

    def SaveImagesSideBySide(self, imgA, imgB, fname,
                                titleA='Original',
                                titleB='Modified',
                                grayscale=False):
        '''
        Display 2 images side by side and save the figure
        for a quick comparison
        '''
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(imgA)
        ax1.set_title(titleA, fontsize=30)
        if grayscale:
            ax2.imshow(imgB, cmap=plt.cm.gray)
        else:
            ax2.imshow(imgB)
        ax2.set_title(titleB, fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(self.outputFolder + fname)

    def Save3ImagesSideBySide(self, imgA, imgB, imgC, fname,
                                titleA='Original Image',
                                titleB='Modified',
                                titleC='Other',
                                grayscale=False,
                                plots=(0,1,1)):
        '''
        Display 3 images side by side and save the figure
        for a quick comparison
        The plots arguments defines whether a plot or imshow needs
        to be called on the corresponding positional arguments
        '''
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(imgA)
        ax1.set_title(titleA, fontsize=30)
        if plots[1]:
            ax2.plot(imgB)
        else:
            if grayscale:
                ax2.imshow(imgB, cmap=plt.cm.gray)
            else:
                ax2.imshow(imgB)
        ax2.set_title(titleB, fontsize=30)

        if plots[2]:
            ax3.plot(imgC)
        else:
            if grayscale:
                ax3.imshow(imgC, cmap=plt.cm.gray)
            else:
                ax3.imshow(imgC)
        ax3.set_title(titleC, fontsize=30)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(self.outputFolder + fname)

    def SaveImage(self, img, fname, grayscale=False):
        '''
        Save individual image
        '''
        if grayscale:
            mpimg.imsave(self.outputFolder + fname, img, cmap=plt.cm.gray)
        else:
            mpimg.imsave(self.outputFolder + fname, img)
        print("Saved : " + fname)

    @staticmethod
    def DrawSlidingBoxes(img, bboxes, color=(0, 0, 255), thick=4, random=False):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        if random:
            for bbox in bboxes:
                if np.random.randint(0, 100) < 25:
                    cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        else:
            for bbox in bboxes:
                # Draw a rectangle given bbox coordinates
                #color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy
