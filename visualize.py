import numpy as np
import matplotlib.pyplot as plt

"""
   Visualize interest point detections on an image.

   Plot N detected interest points as red circles overlaid on the image.

   As a visual indicator of interest point score, scale red color intensity
   with relative score.

   Arguments:
      image       - a grayscale image in the form of a 2D numpy
      xs          - numpy array of shape (N,) containing x-coordinates
      ys          - numpy array of shape (N,) containing y-coordinates
      scores      - numpy array of shape (N,) containing scores
"""
def plot_interest_points(image, xs, ys, scores):
   assert image.ndim == 2, 'image should be grayscale'
   # determine color scale
   s_rank = np.argsort(scores)
   N = s_rank.size
   colors = np.zeros((N,3))
   colors[:,0] = 0.95 * (s_rank / N) + 0.05
   # display points
   plt.figure()
   plt.imshow(image, cmap='gray')
   plt.scatter(ys,xs,c=colors)

"""
   Visualize feature matches.

   Draw lines from feature locations in the first image to matching locations
   in the second.

   Only display matches with scores above a specified threshold (th).

   Reasonable values for the threshold are dependent on your scheme for
   scoring matches.  Varying the threshold to display only the best matches
   can be a useful debugging tool.

   Arguments:
      image0  - a grayscale image in the form of a 2D numpy (first image)
      image1  - a grayscale image in the form of a 2D numpy (second image)
      xs0     - numpy array of shape (N0,) containing x-coordinates of the
                interest points for features in the first image
      ys0     - numpy array of shape (N0,) containing y-coordinates of the
                interest points for features in the first image
      xs1     - numpy array of shape (N1,) containing x-coordinates of the
                interest points for features in the second image
      ys1     - numpy array of shape (N1,) containing y-coordinates of the
                interest points for features in the second image
      matches - a numpy array of shape (N0,) containing, for each feature in
                the first image, the index of the best match in the second
      scores  - a numpy array of shape (N0,) containing a real-valued score
                for each pair of matched features
      th      - threshold; only display matches with scores above threshold
"""
def plot_matches(image0, image1, xs0, ys0, xs1, ys1, matches, scores, th):
   assert image0.ndim == 2, 'image should be grayscale'
   assert image1.ndim == 2, 'image should be grayscale'
   # combine images
   sx0, sy0 = image0.shape
   sx1, sy1 = image1.shape
   sx = sx0 + sx1
   sy = max(sy0, sy1)
   image = np.zeros((sx, sy))
   image[0:sx0,0:sy0]       = image0;
   image[sx0:sx0+sx1,0:sy1] = image1;

   # get coordinates of matches
   xm = xs1[matches]
   ym = ys1[matches]
   # draw correspondence
   plt.figure()
   plt.imshow(image, cmap='gray')
   X = np.zeros((2))
   Y = np.zeros((2))
   N = matches.size
   for n in range(N):
      if (scores[n] > th):
         X[0] = xs0[n]
         X[1] = xm[n]+sx0
         Y[0] = ys0[n]
         Y[1] = ym[n]
         plt.plot(Y,X,'b-')
         plt.plot(Y[0],X[0],'ro')
         plt.plot(Y[1],X[1],'ro')

"""
   Given two images and an translation t = [tx ty] that aligns them, overlay
   and display them in a common coordinate frame.

   The second image is translated and pasted on top of the first.

   Arguments:
      image0  - a grayscale image in the form of a 2D numpy (first image)
      image1  - a grayscale image in the form of a 2D numpy (second image)
      tx      - predicted translation in x-direction between images
      ty      - predicted translation in y-direction between images
"""
def show_overlay(image0, image1, tx, ty):
   assert image0.ndim == 2, 'image should be grayscale'
   assert image1.ndim == 2, 'image should be grayscale'
   # combine images
   sx0, sy0 = image0.shape
   sx1, sy1 = image1.shape
   tx = int(round(tx))
   ty = int(round(ty))
   bx = abs(tx)
   by = abs(ty)
   sx = max(sx0, sx1) + 2 * bx
   sy = max(sy0, sy1) + 2 * by
   image = np.zeros((sx, sy))
   image[bx:sx0+bx,by:sy0+by] = image0;
   image[bx+tx:sx1+bx+tx,by+ty:sy1+by+ty] = image1;
   # draw
   plt.figure()
   plt.imshow(image, cmap='gray')
