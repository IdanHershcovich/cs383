part1.py:

this file contains the math used to compute parts d and e of the theory section.This file can be run independently with no arguments necessary.


HW1.py:

This file contains parts 2 and 3 of HW1.

The yalefaces folder MUST be in the same directory as this file.

to run hw1.py, it is only necessary to call it with "python HW1.py" no arguments necessary as the functions inside the script already have targets to use as args.

For part 2:
on the matrix made of the yalefaces data, it is standardized and then pca is performed on it. All functions were made by hand. PCA package was not used. The part for plotting the 2d data is uncommented. When running HW1.py, the visualization of the data will open on a window, and it must be closed in order to run the rest of the script. The visualization is attached with the PDF  of the HW.

For part 3: 

the only thing done for part 3 is calling the lossyComp function on the standardized yalematrix, using the first image as the image to rebuild, and then a K valulue to represent the PCs. K can be changed in the program to get more or less principal components, if desired.

Running this file (especially lossyComp) will create a .avi file!!!

-Idan Hershcovich
ih64@drexel.edu
