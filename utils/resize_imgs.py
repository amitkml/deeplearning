import glob
from PIL import Image, ImageOps
import os
import cv2
#globbing utility.
import glob
from skimage import io

# %matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import display
from IPython.display import Image as _Imgdis
import os


class ImageProcessor:

    def __init__(self, img_size, resized_dir='RAW_Image_Resized'):
        self.img_size = img_size
        self.resized_dir = resized_dir

    def resize_imgs(self, path):
         """This function reads all images from specified path and resized them. All resized images are kept in resized directory"""
        for i, img_path in enumerate(glob.glob(path)):

            print(i, img_path)
            img = Image.open(img_path)
            img = ImageOps.fit(img, (self.img_size[0], self.img_size[1]), Image.ANTIALIAS)
            name = img_path.split("\\")[-1].split('.')[0]
            if not os.path.exists(f"resized_{self.img_size[0]}_{self.img_size[1]}"):
                os.makedirs(f"resized_{self.img_size[0]}_{self.img_size[1]}")
        img.save(f"resized_{self.img_size[0]}_{self.img_size[1]}/{name}.png")

    def resize_imgs_cv2(self,imagePath,TargetSize):
         """This function reads all images from specified path and resized them by using open CV. All resized images are kept in RAW_Image_Resized directory"""
        !mkdir self.resized_dir
        for bb,file in enumerate (glob.glob(imagePath)):
            img = io.imread(file)
            img = cv2.resize(img, dsize=(TargetSize, TargetSize), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('RAW_Image_Resized/car{}.jpg'.format("{0:04}".format(bb+1)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(1000)
            cv2.destroyAllWindows()

    def read_images_dir(TargetPath):
        """This function reads all images from specified path returns in numpy array"""
        car_images = []

        for bb,file in enumerate (glob.glob(TargetPath)):
            img = io.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            car_images.append(img)

        car_images = np.asarray(car_images)
         return car_images




## imgprocessor = ImageProcessor((256, 192))
## imgprocessor.resize_imgs("data")
