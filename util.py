from scipy import misc
import numpy as np

"""
   Convert an RGB image to grayscale.

   This function applies a fixed weighting of the color channels to form the
   resulting intensity image.

   Arguments:
      rgb   - a 3D numpy array of shape (sx, sy, 3) storing an RGB image

   Returns:
      gray  - a 2D numpy array of shape (sx, sy) storing the corresponding
              grayscale image
"""
def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3],[0.29894, 0.58704, 0.11402])
    return gray

"""
   Load an image and convert it to grayscale.

   Arguments:
      filename - image file to load

   Returns:
      image    - a 2D numpy array containing a grayscale image
"""
def load_image(filename):
   image = misc.imread(filename)
   image = image / 255;
   if (image.ndim == 3):
      image = rgb2gray(image)
   return image
