import os
import numpy as np
import random
# from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
random.seed(0)


def euclidean(centroid, observation):
    distance =math.sqrt(
        ((centroid.item(0)-observation.item(0))**2)
        + ((centroid.item(1)-observation.item(1))**2)
        + ((centroid.item(2)-observation.item(2))**2)
        )
    return distance

def standardize(data):
   stand = (data-data.mean(axis=(0), keepdims=False))/data.std(axis=(0,1), keepdims=False, ddof=1)
   return stand, data.mean(axis=(0), keepdims=False), data.std(axis=(0,1), keepdims=False, ddof=1)

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
    return X_pca


data = np.genfromtxt('diabetes.csv', delimiter=',')

###Fitting the data
class_label_Y = data[:,0]
obs_data_X, x_mean, x_std = standardize(data[:,1:])


def myKMeans(data_clusterX, target_clusterY, k):
    ref_vector = np.empty([0,3], int)
    print(str.format("shape of empty ref vector {}",ref_vector.shape))

    print(str.format("shape of cluster {}",data_clusterX.shape))
    if np.size(data_clusterX,1) > 3:
        data_clusterX = pca(data_clusterX, 3)
        print(str.format("shape of cluster after pca {}",data_clusterX.shape))

    # Get my reference vectors
    for i in range(k):
        rand = random.randrange(0,np.size(data_clusterX,axis=0))
        rand = np.column_stack(data_clusterX[rand])
        # ref_vector = np.append(ref_vector, rand, axis=0)

    ##dictionary for the reference vectors.  The value comes from a random observation in the data.
    centroids = {
        i+1: np.column_stack(data_clusterX[random.randrange(0,np.size(data_clusterX,axis=0))])
        for i in range(k)
    }

    # ##Figure 
    # fig = plt.figure(figsize=(5, 5))
    # colmap = {1: 'r', 2: 'g', 3: 'b'}
    # for i in centroids.keys():
    #     plt.scatter(*centroids[i], color=colmap[i])

    # print(centroids.keys())
    # print(centroids[1].item(0))
    # print(data_clusterX[0].item(0))
    # print(np.size(data_clusterX,axis=0))

    
    
    for obs in range(np.size(data_clusterX,axis=0)):
        min = 1000
        min_index = 0
        for i in centroids.keys():
            dist = euclidean(centroids[i], data_clusterX[obs])
            
            if dist < min: 
               
                min = dist
                min_index = i
                
               
            print(str.format("distance from observation {} to centroid {} is: {}", obs+1 , i, dist))
        print(str.format(" observation {} belongs to centroid {} with distance: {}", obs+1, min_index, min))

        ###Put data in clusters. Each cluster should have the observation that belongs to it. Stuck here
        clusters = {
            min_index: (obs+1,min )
            for i in range(obs)
        }
    print(clusters.items())
    

print(str.format("shape of label {}", class_label_Y.shape))
print(str.format("shape of obs_data {}", obs_data_X.shape))

myKMeans(obs_data_X, class_label_Y, 7)

