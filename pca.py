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

   stand = (data-data.mean(axis=(0,1), keepdims=1))/data.std(axis=(0,1), keepdims=1)
   return stand

def pca(X):
    # Data matrix X, assumes 0-centered
    n, m = X.shape
    # assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n-1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    # Project X onto PC space
    X_pca = np.dot(X, eigen_vecs)
    
    return X_pca


###calls the function that processes the images and puts them in a matrix
matrix_test = yaleMatrix('yalefaces', empty_matrix)

###standardizes the matrix
standardized_matrix = standardize(matrix_test)


###matrix after pca
pcam= pca(standardized_matrix)

plt.scatter(pcam[ : , 1],pcam[ : , 2]) 

plt.show()



