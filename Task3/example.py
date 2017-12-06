import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance

np.random.seed(23)
#num_d2_centers = 200
mapper_coreset_size = 1200
k = 200
k_means_iterations = 10
lambda_val = 5

def sumD2ClusterMemebers(dist_vec, nearest_centers, num_d2_centers):
    summary = []
    for j in range(num_d2_centers):
        clusterID_idx = np.where(nearest_centers == j)[0]
        counts = np.shape(clusterID_idx)[0]
        sum_elements = np.sum(dist_vec[clusterID_idx])
        summary.append(np.append(counts,sum_elements))

    return np.asarray(summary)


def getD2Centers(X, k = 200):

    d2_centers = []

    curr_center = X[np.random.randint(X.shape[0])]
    d2_centers.append(curr_center)
    constant = - 1e10

    dist_curr_center = np.reshape(np.full(X.shape[0], np.iinfo(np.int64).max),(X.shape[0],1))
    #for j in range(num_d2_centers - 1):
    while np.sum(dist_curr_center[:,0]) > constant:
        dist_curr_center = np.minimum(dist_curr_center, cdist(X, [curr_center], 'euclidean'))
        new_center = X[np.random.choice(X.shape[0], p = (dist_curr_center[:,0])/np.sum(dist_curr_center[:,0]))]
        curr_center = new_center
        d2_centers.append(curr_center)
        constant = 16 * lambda_val *  np.shape(d2_centers)[0] * (np.log2(np.shape(d2_centers)[0]) + 2)
    print(np.shape(d2_centers))

    return d2_centers

def impSampling(X,d2_centers):

    dist_d2_centers = cdist(X, d2_centers, 'euclidean') # Distances of all 3000 points in a batch to the d2 centers
    nearest_d2_center = np.argmax(dist_d2_centers, axis = 1) # nearest d2 center for each point

    min_dist = np.amin(dist_d2_centers, axis = 1) # minimum distance to d2 center
    d2_mindist_sum = np.sum(min_dist) #
    c_phi = d2_mindist_sum/X.shape[0]
    B_i = sumD2ClusterMemebers(min_dist, nearest_d2_center, np.shape(d2_centers)[0])
    alpha = 16 * (np.log2(np.shape(d2_centers)[0]) + 2) + 2

    termA = 2 * alpha * min_dist / c_phi

    termB = []
    termC = []
    for j in range(X.shape[0]):
        termB.append((4 * alpha * B_i[nearest_d2_center[j],1])/(c_phi * B_i[nearest_d2_center[j],0]))
        termC.append(4 * X.shape[0]/B_i[nearest_d2_center[j],0])
    q = termA + termB + termC + 1
    prob = q/ np.sum(q)

    return prob

# key: None
# value: 2d numpy array
def mapper(key, value):

    centers = getD2Centers(value)

    importance_prob = impSampling(value,centers)
    imp_samples_idx = np.random.choice(value.shape[0], size = mapper_coreset_size, p = importance_prob)

    for index in imp_samples_idx:
        centers.append(value[index])

    yield 0, centers


# key: key from mapper used to aggregate
# values: list of all value for that key
# Note that we do *not* output a (key, value) pair here.
def reducer(key, values):

    centers = list()
    center_indices = np.random.choice(len(values), k)

    for idx in center_indices:
        centers.append(values[idx])

    centers = np.array(centers)

    for i in xrange(1, k_means_iterations + 1):
        Euclid_distance = cdist(values, centers, 'euclidean')
        min_indices = np.argmin(Euclid_distance, axis=1)

        for clusterID in range(200):
            if(clusterID in min_indices):
                cluster_idx = np.where(min_indices == clusterID)[0]
                centers[clusterID, :] = np.mean(values[cluster_idx, :],axis = 0)

    yield centers


