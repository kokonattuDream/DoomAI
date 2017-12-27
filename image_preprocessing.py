import numpy as np
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box



class PreprocessImage(ObservationWrapper):
    """
    Preprocess the images from Doom
    
    """
    
    def __init__(self, env, height = 64, width = 64, grayscale = True, crop = lambda img: img):
        """
        Intialize the PreprocessImage
        
        @param env: environment (Doom)
        @param height: height of image, default = 64
        @param width: width of image, default = 64
        @param grayscale: black-white image
        @param crop: crop image
        """
        
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop
        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors, height, width])

    def _observation(self, img):
        """
        Observe the image
        
        @param img: input image
        """
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims = True)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype('float32') / 255.
        
        return img
