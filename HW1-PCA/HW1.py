import os
import numpy as np
from PIL import Image as im
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

width = 40
height = 40

empty_matrix = np.empty([0,1600], dtype=np.uint8)


### YALEFACES FOLDER MUST BE IN THE SAME DIRECTORY AS THIS FILE!!!!

###given an image, a height and a width, resize and return the new image
def resize_im(image, h, w):
    resized = image.resize((h,w))
    return resized


###from a given directory, loads all images, reduces size to 40x40, flattens it to a 1d array and then concatenates it to a matrix specified by the user
def yaleMatrix(directory, matrix):
    face_list = os.listdir(directory) #opens the yalefaces folder
    face_list = sorted(face_list)
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
   return stand, data.mean(axis=(0), keepdims=False), data.std(axis=(0,1), keepdims=False, ddof=1)

# Arguments for pca are the matrix with the data, and the number of dimensions we want to reduce to.
def pca(X, dims):
    # Data matrix X, assumes 0-centered
    n, m = X.shape
    assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.cov(X, rowvar=0, ddof=1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)

    # Sorts eigenvalues and eigen vectors. Returns the most signifacnt eigen vals/vec. Quantity of dimensions specified with the dims argument.
    idx = eigen_vals.argsort()[-dims:][::-1]
    eigen_vals = eigen_vals[idx]
    eigen_vecs = eigen_vecs[:,idx]

    # Project X onto PC space
    X_pca = np.dot(-X, eigen_vecs)
    return X_pca, eigen_vals, eigen_vecs


# Function takes 3 params. The matrix, the index of the image you want to apply the function to (the index is used on the standardized matrix)
# and the index K for how many principal components the user wants. 
def lossyComp(data, image_index, k):

    # OpenCV video
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (width, height),isColor=False)

    # Performs pca on the given matrix. Returns eig vals and vecs
    data,eigen_vals, eigen_vecs = pca(data,k)
   
    
    #single image
    image = standardized_matrix[[image_index], :]    

    # for k in D
    for cols in range(k):
        eig_features = eigen_vecs[:,:cols]
        
        ## Projection
        proj = np.matmul(image,eig_features)

        ## Reconstruct
        reconstruction = np.dot(proj, eig_features.transpose())
       

        ## Unstandardize
        reconstruction *= standardized_std
        reconstruction += standardized_mean

        # Resize to 40x40
        reconstruction = reconstruction.reshape((40, 40))
        reconstruction[reconstruction>255] = 255
       
        
        out.write(np.uint8(reconstruction))

    out.release()
    cv2.destroyAllWindows()
    

# Part 2
#Calls the function that processes the images and puts them in a matrix
yaleMatrix = yaleMatrix('yalefaces', empty_matrix)

# Standardizes the matrix
standardized_matrix, standardized_mean, standardized_std = standardize(yaleMatrix)


# standardized yalefaces Matrix after pca
matrix_pca, eigen_vals, eigen_vecs= pca(standardized_matrix, 2)




# ## Visualization

#Once the image shows when the script is called, the image viewer must be closed in order to run the rest of the script!
plt.scatter(matrix_pca[ : , 0],matrix_pca[ : , 1]) 

plt.savefig('2d_plot.png')


### Part 3
# Calls lossy compresison on the standardized yalematrix 

lossyComp(standardized_matrix, 0, 1600)




    