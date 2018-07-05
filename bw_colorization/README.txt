The file requires Python and the following libraries: OpenCV, cvxopt, scipy.

Line 45 should contain the grayscale image.
Line 46 should contain the mockup image.

In this project, we consider algorithms to add coloring to monochrome images. The implemented algorithms rely on the assumption that neighboring pixels with similar monochrome intensities will have similar colors.

Algorithm implemented:
http://webee.technion.ac.il/people/anat.levin/papers/colorization-siggraph04.pdf

RGB<->YIV conversion taken from:
https://github.com/asafdav2/colorization_using_optimization/blob/master/color_conv.py
