Inside the folders the image file names are composed of:

imgNumber-OutputType.png

Output Types:
-gray: Gray gaussian filter smoothed image (size=9,sigma=2)
-trunc: The thresholded image using opencv THRESH_TRUNC with computed and shifted otsu threshold
-hough: Iris detection with canny and hough cirlce transformation & incrementing circle to find start of sclera
-normIris: The gray iris transformed into a 2D matrix (w:360, h:128)
-ordinalIris: Ordinal intensity comparison of horizontal layers (A < B ? 1 : 0)
-ordinalIris_vertical: Same as above but for vertical layers
-ordinalIris8: Using LBP, so eight neighbours of each pixel and do ordinal intensity comparison