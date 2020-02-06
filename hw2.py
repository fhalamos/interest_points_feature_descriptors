import numpy as np
from canny import *

"""
   INTEREST POINT OPERATOR (12 Points Implementation + 3 Points Write-up)

   Implement an interest point operator of your choice.

   Your operator could be:

   (A) The Harris corner detector (Szeliski 4.1.1)

               OR

   (B) The Difference-of-Gaussians (DoG) operator defined in:
       Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", 2004.
       https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

               OR

   (C) Any of the alternative interest point operators appearing in
       publications referenced in Szeliski or in lecture

              OR

   (D) A custom operator of your own design

   You implementation should return locations of the interest points in the
   form of (x,y) pixel coordinates, as well as a real-valued score for each
   interest point.  Greater scores indicate a stronger detector response.

   In addition, be sure to apply some form of spatial non-maximum suppression
   prior to returning interest points.

   Whichever of these options you choose, there is flexibility in the exact
   implementation, notably in regard to:

   (1) Scale

       At what scale (e.g. over what size of local patch) do you operate?

       You may optionally vary this according to an input scale argument.

       We will test your implementation at the default scale = 1.0, so you
       should make a reasonable choice for how to translate scale value 1.0
       into a size measured in pixels.

   (2) Nonmaximum suppression

       What strategy do you use for nonmaximum suppression?

       A simple (and sufficient) choice is to apply nonmaximum suppression
       over a local region.  In this case, over how large of a local region do
       you suppress?  How does that tie into the scale of your operator?

   For making these, and any other design choices, keep in mind a target of
   obtaining a few hundred interest points on the examples included with
   this assignment, with enough repeatability to have a large number of
   reliable matches between different views.

   If you detect more interest points than the requested maximum (given by
   the max_points argument), return only the max_points highest scoring ones.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      image       - a grayscale image in the form of a 2D numpy array
      max_points  - maximum number of interest points to return
      scale       - (optional, for your use only) scale factor at which to
                    detect interest points

   Returns:
      xs          - numpy array of shape (N,) containing x-coordinates of the
                    N detected interest points (N <= max_points)
      ys          - numpy array of shape (N,) containing y-coordinates
      scores      - numpy array of shape (N,) containing a real-valued
                    measurement of the relative strength of each interest point
                    (e.g. corner detector criterion OR DoG operator magnitude)

  Reference: https://muthu.co/harris-corner-detector-implementation-in-python/
"""
def find_interest_points(image, max_points = 200, scale = 1.0):
  # check that image is grayscale
  assert image.ndim == 2, 'image should be grayscale'

  #Harris corner detector

  #1. Spatial derivative calculation
  I_x, I_y = sobel_gradients(image)

  #2. Matrix of second derivatives

  # The classic “Harris” detector (Harris and Stephens
  # 1988) uses a [-2 -1 0 1 2] filter, but more modern variants (Schmid, Mohr, and Bauckhage
  # 2000; Triggs 2004) convolve the image with horizontal and vertical derivatives of a Gaussian
  # (typically with σ = 1)

  #---> Im not fully getting this step
  Ixx = conv_2d_gaussian(I_x**2)
  Ixy = conv_2d_gaussian(I_y*I_x)
  Iyy = conv_2d_gaussian(I_y**2)

  #3. Compute corner response (R)
  alpha = 0.05

  detM = Ixx * Iyy - Ixy ** 2
  traceM = Ixx + Iyy

  R = detM - alpha*traceM**2

  # NOTES:
  # "k(alpha) is the sensitivity factor to separate corners from edges,
  # typically a value close to zero, for my analysis I have taken k=0.04.
  # Small values of k result in detection of sharp corners.
  # The response R is an array of peak values of each row in the image.
  # We can use these peak values to isolate corners and edges as I will
  # describe in the next step."


  #4. We use a gaussian filter and nonmaxsupression to eliminate corner detections that are not strong enough compared to neighbors
  R = conv_2d_gaussian(R)
  image_corners_nm,_ = canny_nmax(R)

  xs = np.empty(1)
  ys = np.empty(1)
  scores = np.empty(1)

  #5. Find corners using R > 0
  for rowindex, response in enumerate(image_corners_nm  ):
    for colindex, r in enumerate(response):
        if r > 0:
          # image_corners[rowindex, colindex] = r
          xs = np.concatenate((xs , [rowindex])) # This might be a quite inefficient way of a "dynamic" numpy array
          ys = np.concatenate((ys , [colindex]))
          scores = np.concatenate((scores , [r]))

  #6. Keep top max_points 
  #If length of identified points is > max_points, we should return the highest scores
  if(xs.size>max_points):
    top_scores_indices = scores.argsort()[-max_points:][::-1]
    xs = np.asarray([xs[i] for i in top_scores_indices])
    ys = np.asarray([ys[i] for i in top_scores_indices])
    scores = np.asarray([scores[i] for i in top_scores_indices])
  return xs, ys, scores


'''
Returns array of length 8, indication energy of each orientation
The energy of each orientation equals the sum of the magnitudes of the gradients that 
fall in the given orientation.
'''
def get_gradient_histogram_of_window(mag, theta):
  energy = np.zeros(8)

  for row_i, row_theta in enumerate(theta):
    for col_i, angle in enumerate(row_theta):

      angle_translated = angle + np.pi #angles now from 0 to 2pi
      list_index = int(angle_translated/(np.pi/4)) if int(angle_translated/(np.pi/4)) != 8 else 0
      energy[list_index]+= mag[row_i, col_i]

  return energy



"""
   FEATURE DESCRIPTOR (12 Points Implementation + 3 Points Write-up)

   Implement a SIFT-like feature descriptor by binning orientation energy
   in spatial cells surrounding an interest point.

   Unlike SIFT, you do not need to build-in rotation or scale invariance.

   A reasonable default design is to consider a 3 x 3 spatial grid consisting
   of cell of a set width (see below) surrounding an interest point, marked
   by () in the diagram below.  Using 8 orientation bins, spaced evenly in
   [-pi,pi), yields a feature vector with 3 * 3 * 8 = 72 dimensions.

             ____ ____ ____
            |    |    |    |
            |    |    |    |
            |____|____|____|
            |    |    |    |
            |    | () |    |
            |____|____|____|
            |    |    |    |
            |    |    |    |
            |____|____|____|

                 |----|
                  width

   You will need to decide on a default spatial width.  Optionally, this can
   be a multiple of a scale factor, passed as an argument.  We will only test
   your code by calling it with scale = 1.0.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

  Arguments:
      image    - a grayscale image in the form of a 2D numpy
      xs       - numpy array of shape (N,) containing x-coordinates
      ys       - numpy array of shape (N,) containing y-coordinates
      scale    - scale factor

   Returns:
      feats    - a numpy array of shape (N,K), containing K-dimensional
                 feature descriptors at each of the N input locations
                 (using the default scheme suggested above, K = 72)
"""
def extract_features(image, xs, ys, scale = 1.0):
  # check that image is grayscale
  assert image.ndim == 2, 'image should be grayscale'

  image = mirror_border(image, wx = 4, wy = 4)

  dx, dy = sobel_gradients(image)
  mag   = np.sqrt((dx * dx) + (dy * dy))
  theta = np.arctan2(dy, dx)
    
  feats = np.empty([len(xs),72])

  #For each of the interest points, capture histogram of gradients around it
  for interest_point_index, px in enumerate(xs):
    
    px = px + 4 #Considering we added padding
    py = ys[interest_point_index] + 4

    #For each of the 9 neighboring windows
    deviations_x = [-3,0,3]
    deviations_y = [-3,0,3]

    window_counter=0
    for dev_x in deviations_x:
      for dev_y in deviations_y:
        wc_x = int(px+dev_x)
        wc_y = int(py+dev_y)

        #Extract region of interest
        mag_roi = mag[wc_x-1:wc_x+2, wc_y - 1:wc_y + 2]
        theta_roi = theta[wc_x-1:wc_x+2, wc_y - 1:wc_y + 2]

        window_histogram = get_gradient_histogram_of_window(mag_roi, theta_roi)

        #Save feature
        for i in range(0,8): 
          feats[interest_point_index,window_counter*8+i] = window_histogram[i]

        window_counter+=1

  return feats


def chiSquared(p,q):
    return 0.5*np.sum((p-q)**2/(p+q+1e-6))

"""
   FEATURE MATCHING (7 Points Implementation + 3 Points Write-up)

   Given two sets of feature descriptors, extracted from two different images,
   compute the best matching feature in the second set for each feature in the
   first set.

   Matching need not be (and generally will not be) one-to-one or symmetric.
   Calling this function with the order of the feature sets swapped may
   result in different returned correspondences.

   For each match, also return a real-valued score indicating the quality of
   the match.  This score could be based on a distance ratio test, in order
   to quantify distinctiveness of the closest match in relation to the second
   closest match. It could optionally also incorporate scores of the interest
   points at which the matched features were extracted.  You are free to
   design your own criterion. Note that you are required to implement the naive
   linear NN search. For 'lsh' and 'kdtree' search mode, you could do either to
   get full credits.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices. You are required to report the efficiency comparison
   between different modes by measure the runtime (check the benchmarking related
   codes in hw2_example.py).

   Arguments:
      feats0   - a numpy array of shape (N0, K), containing N0 K-dimensional
                 feature descriptors (generated via extract_features())
      feats1   - a numpy array of shape (N1, K), containing N1 K-dimensional
                 feature descriptors (generated via extract_features())
      scores0  - a numpy array of shape (N0,) containing the scores for the
                 interest point locations at which feats0 was extracted
                 (generated via find_interest_point())
      scores1  - a numpy array of shape (N1,) containing the scores for the
                 interest point locations at which feats1 was extracted
                 (generated via find_interest_point())
      mode     - 'naive': performs a brute force NN search

               - 'lsh': Implementing the local senstive hashing (LSH) approach
                  for fast feature matching. In LSH, the high dimensional
                  feature vectors are randomly projected into low dimension
                  space which are further binarized as boolean hashcodes. As we
                  group feature vectors by hashcodes, similar vectors may end up
                  with same 'bucket' with high propabiltiy. So that we can
                  accelerate our nearest neighbour matching through hierarchy
                  searching: first search hashcode and then find best
                  matches within the bucket.
                  Advice for impl.:
                  (1) Construct a LSH class with method like
                  compute_hash_code   (handy subroutine to project feature
                                      vector and binarize)
                  generate_hash_table (constructing hash table for all input
                                      features)
                  search_hash_table   (handy subroutine to search hash table)
                  search_feat_nn      (search nearest neighbour for input
                                       feature vector)
                  (2) It's recommended to use dictionary to maintain hashcode
                  and the associated feature vectors.
                  (3) When there is no matching for queried hashcode, find the
                  nearest hashcode as matching. When there are multiple vectors
                  with same hashcode, find the cloest one based on original
                  feature similarity.
                  (4) To improve the robustness, you can construct multiple hash tables
                  with different random project matrices and find the closest one
                  among all matched queries.
                  (5) It's recommended to fix the random seed by random.seed(0)
                  or np.random.seed(0) to make the matching behave consistenly
                  across each running.

               - 'kdtree': construct a kd-tree which will be searched in a more
                  efficient way. https://en.wikipedia.org/wiki/K-d_tree
                  Advice for impl.:
                  (1) The most important concept is to construct a KDNode. kdtree
                  is represented by its root KDNode and every node represents its
                  subtree.
                  (2) Construct a KDNode class with Variables like data (to
                  store feature points), left (reference to left node), right
                  (reference of right node) index (reference of index at original
                  point sets)and Methods like search_knn.
                  In search_knn function, you may specify a distance function,
                  input two points and returning a distance value. Distance
                  values can be any comparable type.
                  (3) You may need a user-level create function which recursively
                  creates a tree from a set of feature points. You may need specify
                  a axis on which the root-node should split to left sub-tree and
                  right sub-tree.


   Returns:
      matches  - a numpy array of shape (N0,) containing, for each feature
                 in feats0, the index of the best matching feature in feats1
      scores   - a numpy array of shape (N0,) containing a real-valued score
                 for each match
"""
def match_features(feats0, feats1, scores0, scores1, mode='naive'):
  
  n_features_0, k = feats0.shape

  matches = np.zeros(n_features_0)#, dtype=np.int8)#, dtype=float64)#, dtype=np.int8)
  scores = np.zeros(n_features_0)#, dtype=np.float64)#, dtype=np.int8)

  for index_0, f_0 in enumerate(feats0):
    n_features_1, _ = feats1.shape
    distances_to_f1 = np.empty(n_features_1)

    for index_1, f_1 in enumerate(feats1):
      distances_to_f1[index_1] = chiSquared(f_0,f_1)
 
    index_best, index_second_best = distances_to_f1.argsort()[0:2]
    matches[index_0] = index_best
    scores[index_0] = distances_to_f1[index_second_best]/distances_to_f1[index_best]

  matches = matches.astype(int)

  return matches, scores

"""
   HOUGH TRANSFORM (7 Points Implementation + 3 Points Write-up)

   Assuming two images of the same scene are related primarily by
   translational motion, use a predicted feature correspondence to
   estimate the overall translation vector t = [tx ty].

   Your implementation should use a Hough transform that tallies votes for
   translation parameters.  Each pair of matched features votes with some
   weight dependant on the confidence of the match; you may want to use your
   estimated scores to determine the weight.

   In order to accumulate votes, you will need to decide how to discretize the
   translation parameter space into bins.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
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

   Returns:
      tx      - predicted translation in x-direction between images
      ty      - predicted translation in y-direction between images
      votes   - a matrix storing vote tallies; this output is provided for
                your own convenience and you are free to design its format
"""
def hough_votes(xs0, ys0, xs1, ys1, matches, scores):
   ##########################################################################
   # TODO: YOUR CODE HERE
   raise NotImplementedError('hough_votes')
   ##########################################################################
   return tx, ty, votes
