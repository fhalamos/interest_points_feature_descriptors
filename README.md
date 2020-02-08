Design and implement interest point detection, feature descriptor extraction, and feature matching. From matched descriptors in two different views of the same scene, you will predict a transformation (here simplified to translation only) relating the two views.

As interest points, feature descriptors, and matching criterion all involve design choices, there is flexibility in the implementation details, and many reasonable solutions. Note that, in match_features, your implementation will include (1) Naive linear; AND either (2) LSH or (3) KD Tree for nearest neighbor search. (1) is required, as is at least one of (2) or (3).  You can do either (2) or (3) to get full credit.

As was the case for homework 1, your implementation should be restricted to using low-level primitives in numpy.

See hw2.py for detailed descriptions of the functions you must implement.

See hw2_examples.py for testing your code.

In addition to submitting your implementation in hw2.py, you will also need to submit a writeup in the form of a PDF (hw2.pdf) which briefly explains the design choices you made and any observations you might have on their effectiveness. You will also report an efficiency comparison between different modes by measuring the runtime required for feature matching.

Resources:

A working implementation of Canny edge detection is available (canny.py) for your use, should you wish to build your feature descriptor based on Canny edges. Note that the output is changed slightly: this detector returns a nonmax-suppressed edge map, and edge orientations.  Alternatively, you can use your own implementation of Canny edges from homework 1. If so, feel free to replace the provided canny.py with your own.
Some functions for visualizing interest point detections and feature matches are provided (visualize.py).
Submit:

hw2.py - your implementation of homework 2
hw2.pdf - your write-up describing your design choices (Please show the visulizations of the interest points, feature matching,  images overlay, time benchmarking for image retrieval and any other interesting things you have observed. A simple demo is given below).
canny.py (optional) - submit only if you modify or replace the provided implementation