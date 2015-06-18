Pixelwise Geometric Median
============================

Average multiple images, using [geometric median](https://en.wikipedia.org/wiki/Geometric_median).
Averaging is done pixelwise, using iterative vaiation of the Weiszfeld's algorithm.

Usage
-----
geometric_median.py [-h] [-g] [-p] [-t N] [-i N] [--bias BIAS] [-n ORDER] [-c NAME] [-o OUTPUT] IMAGE [IMAGE ...]

*  -h, --help            show this help message and exit
*  -g, --glob            Apply glob on given paths (useful on windows)
*  -p, --preload         Preload all images to memory. Faster, but requires a lot of memory.
*  -t N, --threads N     Use N worker threads, default is processor number+1
*  -i N, --iterations N  Number of iterations of the algorithm
*  --bias BIAS           Bias of the algorithm, small positive floating-point value. Default is 1e-6.
*  -n ORDER, --norm ORDER Order of the norm function to calculate distances. Possible values are 1, 2, inf, -inf
*  -c NAME, --colormap NAME Apply specified color mapping before averaging images. Possible values are: none (use RGB as is), projective[:luma-weight=0.3] (suppress importance of luminance by luma-weight factor, value between 0 and 1)
*  -o OUTPUT, --output OUTPUT Output image file

Requirements
------------

Code written in Python 3, but should be easily adaptable to Python 2. Requires following Python modules:
* Numpy: for numeric calculations
* PIL (or its fork Pillow): for reading and writing image files.

Troubleshooting
---------------

### Script uses too much memory
Try reduce number of worker threads (default is number of processors). 
