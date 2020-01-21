import os
import numpy as np
from PIL import Image as im

import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf) #print all of output without trunc
width = 40
height = 40

empty_matrix = np.empty([0,1600], dtype=np.uint8)




###given an image, a height and a width, resize and return the new image
def resize_im(image, h, w):
    resized = image.resize((h,w))
    return resized


###from a given directory, loads all images, reduces size to 40x40, flattens it to a 1d array and then concatenates it to a matrix specified by the user
def yaleMatrix(directory, matrix):
    face_list = os.listdir(directory) #opens the yalefaces folder
    for entry in face_list:
        og = im.open(directory+'/'+entry)
        res_im = resize_im(og, height, width)
        im_as_arr = np.asarray(res_im, dtype=np.uint8)
        flat = im_as_arr.flatten()
        flat = np.column_stack(flat)
        matrix = np.concatenate((matrix, flat), axis=0)
       
    return matrix

###standardizing data formula is Z = (X-meanOfDimension) / std. X is the current value of the matrix that is going to be standardized.
def standardize(data):

   stand = (data-data.mean(axis=(0), keepdims=False))/data.std(axis=(0,1), keepdims=False, ddof=1)
   return stand

# Arguments for pca are the matrix with the data, and the number of dimensions we want to reduce to.
def pca(X, dims):
    # Data matrix X, assumes 0-centered
    n, m = X.shape
    assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.cov(X, rowvar=0)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    print (str.format("eig vals {}", eigen_vals))

    # Sorts eigenvalues and eigen vectors. Returns the most signifacnt eigen vals/vec. Quantity of dimensions specified with the dims argument.
    idx = eigen_vals.argsort()[-dims:][::-1]
    eigen_vals = eigen_vals[idx]
    eigen_vecs = eigen_vecs[:,idx]

    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs)
    return X_pca
    # X_pca[:,:2] returns first two columns

# Part 2
#Calls the function that processes the images and puts them in a matrix
yaleMatrix = yaleMatrix('yalefaces', empty_matrix)

# Standardizes the matrix
standardized_matrix = standardize(yaleMatrix)


#Matrix after pca
matrix_pca= pca(standardized_matrix, 0)

print(matrix_pca.shape)

## Visualization
# plt.scatter(matrix_pca[ : , 0],matrix_pca[ : , 1]) 

# plt.show()


### Part 3
face = im.open('yalefaces/subject02.centerlight')

res_im = resize_im(face, height, width)
im_as_arr = np.asarray(res_im, dtype=np.uint8)
flat = im_as_arr.flatten()
flat = np.column_stack(flat)
matrix = np.concatenate((empty_matrix, flat), axis=0)

print(matrix.shape)

# for k in D:
    