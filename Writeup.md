# INTEREST POINT OPERATOR

- I used Harris detector
- Gaussians to create second derivative matrix
  # The classic “Harris” detector (Harris and Stephens
  # 1988) uses a [-2 -1 0 1 2] filter, but more modern variants (Schmid, Mohr, and Bauckhage
  # 2000; Triggs 2004) convolve the image with horizontal and vertical derivatives of a Gaussian
  # (typically with σ = 1)



-We apply a gaussian before non-max to kill local max that are too close to each other. So as to have only one.

-A simple (and sufficient) choice is to apply nonmaximum suppression
 over a local region.  In this case, over how large of a local region do
 you suppress?  How does that tie into the scale of your operator?

   For making these, and any other design choices, keep in mind a target of
   obtaining a few hundred interest points on the examples included with
   this assignment, with enough repeatability to have a large number of
   reliable matches between different views.


# FEATURE DESCRIPTOR 

I used histogram of gradients for feature descriptor.
For every point of interest, i look at the 3x3 spacial grid (window) around it, each of them of width 3. For each window, i created a length 8 histogram, one for each pi/4th theta segment, and saved the magnitud of the gradients that fall in each of the theta segments.
-had to use padding so as to capture windowss

#MATCHES

##Navie
-used chisquare in naive, better for distances between distribution


##LSH
->"In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices. You are required to report the efficiency comparison
   between different modes by measure the runtime (check the benchmarking related
   codes in hw2_example.py)."

- found that hash codes of length 20 gave great results

(4) To improve the robustness, you can construct multiple hash tables
with different random project matrices and find the closest one
among all matched queries. My performance was degraded a lot when doing this.

# HOUGH TRANSFORM

The goal of this method is to find the 'most representative' translational vector.
Given that there are many different translational vector between two images, the approach used consists in building a grid of transitional vectors (similar transitional vectors will end up in the same cell in the grid), and each cell will consist of the sum of the scores of the transitional vectors on it.

In the x-axis, the grid contains all possible values of transitional vectors in the x direction. Analogous is done for y-axis.

My approach was to build a grid, where each cell location represented a particular translation (a value of 5 in x axis represented translations of 5 in the x_domain). The value of the cells in the grid where the sum the scores of all translations that ended up in that cell.