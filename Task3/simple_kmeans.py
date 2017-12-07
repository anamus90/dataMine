import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans

np.random.seed(23)

k = 200
rand_restarts = 1
num_rand_samples = 1500
lambda_val = 10

def getD2Centers(X):

    curr_center = X[np.random.randint(X.shape[0])]
    d2_centers = np.array([curr_center])
    # constant = -1e10

    dist_curr_center = np.zeros((X.shape[0]))

    # d2_centers = set()
    # d2_centers.add(np.random.randint(X.shape[0]))
    for j in range(k - 1):
    #while np.sum(dist_curr_center) > constant:
        dist_curr_center = np.amin(cdist(X, d2_centers, 'sqeuclidean'), axis = 1)
        new_center = X[np.random.choice(X.shape[0], p = (dist_curr_center)/np.sum(dist_curr_center))]
        curr_center = new_center
        d2_centers = np.vstack((d2_centers, curr_center))
        #d2_centers.append(curr_center)
        constant = 16 * lambda_val *  np.shape(d2_centers)[0] * (np.log2(np.shape(d2_centers)[0]) + 2)
    print(np.shape(d2_centers))

    return d2_centers

def mapper(key, value):

    rand_idx = np.random.randint(0, value.shape[0], num_rand_samples)
    new_values = value[rand_idx, :]

    yield 0, new_values

def reducer(key, values):

    best_cost =  np.inf
    best_centers = np.random.randn(200, 250)

    for i in range(rand_restarts):
        centers = getD2Centers(values)
        new_centers, new_cost = kmeans(values, centers, thresh = 1e-5)


        if new_cost < best_cost:
            best_cost = new_cost
            best_centers = new_centers

    yield best_centers
    #yield np.random.randn(200, 250)
