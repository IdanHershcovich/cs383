import os
import numpy as np

import matplotlib.pyplot as plt

###standardizing data formula is Z = (X-meanOfDimension) / std. X is the current value of the matrix that is going to be standardized.
def standardize(data):

   stand = (data-data.mean(axis=(0), keepdims=False))/data.std(axis=(0,1), keepdims=False, ddof=1)
   return stand

### PART 1 THEORY PART D and E 
partOneMatrix = np.array([[-2,1],[-5,-4], [-3,1], [0,3], [-8,11], [-2,5], [1,0], [5,-1], [-1,-3], [6,1]])


### Part D. Get the principal components

# Standardize data
stand = standardize(partOneMatrix)

# Compute covariance matrix
C = np.cov(stand, rowvar=0)

# Sorts eigenvalues and eigen vectors. Returns the most signifacnt eigen vals/vec. Quantity of dimensions specified with the dims argument.

eigen_vals, eigen_vecs = np.linalg.eig(C)
idx = eigen_vals.argsort()[-1:][::-1]
eigen_vals = eigen_vals[idx]
eigen_vecs = eigen_vecs[:,idx]
print (str.format("eig vals {}", eigen_vals))

print (str.format("eig vec {}", eigen_vecs))

### Part E. 

# Project X onto 1D space
projected = np.dot(stand, eigen_vecs)
print (str.format("projected vals {}", projected))

print (str.format("proj shape {}", projected.shape))