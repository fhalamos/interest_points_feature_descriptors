Copyright (c) 2010 Adobe Systems, Inc.

Adobe Panoramas Data Set
Contact: Jonathan Brandt (jbrandt@adobe.com)

License:
--------

    This data set may be used subject to the terms described in the accompanying
    file named "license.txt".

Overview:
---------

    This data set consists of a set of 10 panorama image sets, together with 
    precomputed local features for each image, and ground truth homographies for 
    each overlapping image pair.  The purpose of the data set is to test 
    algorithms for local feature matching in the context of image registration.

    If you use these data in a publication, please cite:

    Brandt, J.  "Transform coding for fast approximate nearest neighbor search in
        high dimensions", IEEE Conf. on Computer Vision and Pattern Recognition, 2010.

Contents:
---------

    readme.txt:     This file.
    license.txt:    License file.
    code:           Directory containing MATLAB utilities to access the data.
    data:           The panorama sets (see below).
    setup.m:        MATLAB script to add the code directory to the MATLAB script path.
    demo.m:         Simple demo using the utilities.

Panorama Sets:
--------------

    The data set consists of 10 panorama sets each contained in a subdirectory of the 
    "data" directory.  Each panorama set consists of the following:

    (1) A set of 8 bit grayscale images numbered sequentially, stored both as PNG,
        and as an unformatted raster of bytes (with .raw suffix).
    (2) A set of files containing the local features for each image in text format.
        Each row of the feature file is a space-separated sequence of numbers, the first
        two are the x and y coordinates of the feature, and the remaining 128 numbers are
        the descriptor component values.
    (3) A set of homographies for each overlapping pair of images, named Hnntomm, where
        nn and mm are the indices of each of the images.  Each homography is specified
        with a 3x3 matrix stored in text format.  For instance, if H = H01to00, then

            x0 = (H(1,1)*x1 + H(1, 2)*y1 + H(1,3)) / (H(3,1)*x1 + H(3, 2)*y1 + H(3,3))
            y0 = (H(2,1)*x1 + H(2, 2)*y1 + H(2,3)) / (H(3,1)*x1 + H(3, 2)*y1 + H(3,3))

        where (x1, y1) and (x0, y0) are corresponding points in image 1 and image 0, 
        respectively.

A number of utilities are provided with this data set to load the images, features,
and homographies, and map the features and images subject to the homographies.  
(See MATLAB Utilities below for details.)

The panorama sets are:

    carmel:         18 images, 683 x 1024, 47 pairs
    diamondhead:    23 images, 683 x 1024, 59 pairs
    fishbowl:       13 images, 600 x 900, 35 pairs
    goldengate:     6 images, 900 x 600, 9 pairs
    halfdome:       14 images, 683 x 1024, 51 pairs
    hotel:          8 images, 768 x 1024, 9 pairs
    office:         4 images, 864 x 1152, 6 pairs
    rio:            56 images, 768 x 1024, 426 pairs
    shanghai:       30 images, 683 x 1024, 230 pairs
    yard:           9 images, 728 x 1092, 19 pairs

MATLAB Utilities:
-----------------

    A number of utilities are provided to access and manipulate the data.  For 
    details see help on each individual utility.

    all_pairs:          list all overlapping image pairs for a given panorama set
    apply_homography:   apply the homography to a set of (x, y) points
    check_pair:         visually check a registered image pair and mapped features
    load_image:         load a specified image from a panorama set
    load_pair:          load an overlapping image pair, along with the features and homography
    map_image:          map one image to another subject to given homography
    path_tail:          return a filesystem path tail

Acknowledgement:
----------------

    I am very grateful to Hailin Jin for kindly providing the source images and 
    the use of his SIFT implementation and image registration library to 
    estimate the homographies.

References:
-----------

    [1] Brandt, J.  "Transform coding for fast approximate nearest neighbor search in
        high dimensions", IEEE Conf. on Computer Vision and Pattern Recognition, 2010.
