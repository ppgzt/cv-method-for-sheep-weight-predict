"""
Module responsible for implementing the preprocessing transformations applied in the experiment.
"""
import tensorflow as tf
import numpy as np

"""
Transforms the shape from (240, 320, 1) to (240, 320, 3), only replicating the channel content.
"""
class Replicate1DtoNDimChannel:

    def __init__(self, dim: int):
        self.dim = dim

    def transform(self, data: np.array):
        img_rows, img_cols = data.shape
        g = np.zeros((img_rows, img_cols, self.dim))

        for i in range(0, data.shape[0]):
          for j in range(0, data.shape[1]):
            g[i,j] = [data[i,j]] * self.dim           

        return g
        
"""
Change all pixels with values ​​greater than 1950 to the value 1950 (which corresponds to the node height). Values ​​greater than this threshold are likely noise.
"""
class NoiseRemovalSetMaxValue:

    def __init__(self, max_value: int):
        self.max_value = max_value

    def transform(self, data: np.array):
        g = np.copy(data)

        for i in range(0, data.shape[0]):
          for j in range(0, data.shape[1]):
            pos = (i,j)
            if data[pos] >= self.max_value:
                g[pos] = self.max_value

        return g

"""
Adjust the scale to 0-1, basically dividing the value of each pixel by 1950 (which is the maximum value).
"""
class AdjustScaleWithFixedMaxValue:

    def __init__(self, max_value: int):
        self.max_value = max_value

    def transform(self, data: np.array):
        data = data.astype('float32')
        data /= self.max_value
        
        return data

"""
Resize the image to (300, 300), using the **resize-with-padding** strategy;
"""
class ResizeImageWithPadding:

    def __init__(self, shape: tuple):
        self.shape = shape

    def transform(self, data: np.array):    
        img = tf.image.convert_image_dtype(data, tf.float32)
    
        # Resizes while maintaining proportions (shorter side adjusted)
        resized_img = tf.image.resize_with_pad(img, self.shape[0], self.shape[1])
        
        return resized_img

class FlipImageHorizontally:

    def transform(self, data: np.array):
        return np.fliplr(data)

class FlipImageVertically:

    def transform(self, data: np.array):
        return np.flipud(data)        