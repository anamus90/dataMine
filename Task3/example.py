import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans

np.random.seed(23)

#num_d2_centers = 200
coreset_size = 1500
k = 200
k_means_iterations = 10
lambda_val = 300 # D2 centers proportional to 1/lambda

def sumD2ClusterMemebers(dist_vec, nearest_centers, num_d2_centers):
    summary = [0] * num_d2_centers
    for j in range(num_d2_centers):
        clusterID_idx = np.where(nearest_centers == j)[0]
        counts = np.shape(clusterID_idx)[0]
        sum_elements = np.sum(dist_vec[clusterID_idx])
        summary[j] = np.append(counts,sum_elements)

    return np.asarray(summary)


def getD2Centers(X, k = 200):

    curr_center = X[np.random.randint(X.shape[0])]
    d2_centers = np.array([curr_center])
    constant = -1e10

    dist_curr_center = np.zeros((X.shape[0]))

    #for j in range(num_d2_centers - 1):
    while np.sum(dist_curr_center) > constant:
        dist_curr_center = np.amin(cdist(X, d2_centers, 'sqeuclidean'), axis = 1)
        new_center = X[np.random.choice(X.shape[0], p = (dist_curr_center)/np.sum(dist_curr_center))]
        curr_center = new_center
        d2_centers = np.vstack((d2_centers, curr_center))
        #d2_centers.append(curr_center)
        constant = 16 * lambda_val *  np.shape(d2_centers)[0] * (np.log2(np.shape(d2_centers)[0]) + 2)
    print(np.shape(d2_centers))

    return d2_centers

def impSampling(X,d2_centers):

    alpha = 16 * (np.log2(np.shape(d2_centers)[0]) + 2) + 2

    dist_d2_centers = cdist(X, d2_centers, 'sqeuclidean') # Distances of all 3000 points in a batch to the d2 centers
    nearest_d2_center = np.argmax(dist_d2_centers, axis = 1) # nearest d2 center for each point
    min_dist = np.amin(dist_d2_centers, axis = 1) # minimum distance to d2 center
    d2_mindist_sum = np.sum(min_dist) # sum of min distances
    c_phi = d2_mindist_sum/X.shape[0] # mean cost

    B_i = sumD2ClusterMemebers(min_dist, nearest_d2_center, np.shape(d2_centers)[0])

    termA = 2 * alpha * min_dist / c_phi

    termB = []
    termC = []
    for j in range(X.shape[0]):
        termB.append((4 * alpha * B_i[nearest_d2_center[j],1])/(c_phi * B_i[nearest_d2_center[j],0]))
        termC.append(4 * X.shape[0]/B_i[nearest_d2_center[j],0])
    q = termA + termB + termC + 1
    prob = q/ (np.sum(q))

    weight = 1/(q * coreset_size)
    return np.asarray(prob), weight

# key: None
# value: 2d numpy array
def mapper(key, value):

    centers = getD2Centers(value)

    importance_prob,weights = impSampling(value,centers)
    imp_samples_idx = np.random.choice(value.shape[0], size = coreset_size, p = importance_prob)

    #centers = np.column_stack((np.ones(np.shape(centers)[0]),centers))
    new_values = []

    for index in imp_samples_idx:
        new_values.append(np.hstack((weights[index],value[index])))

    #print(np.shape(new_values))

    yield 0, new_values



# key: key from mapper used to aggregate
# values: list of all value for that key
# Note that we do *not* output a (key, value) pair here.
def reducer(key, values):

    #centers = kmeans(values[:,1:], k)[0]
    centers = list()
    weights = values[:, 0]
    center_indices = np.random.choice(len(values), k, replace = False)

    for idx in center_indices:
       centers.append(values[idx, 1:])

    centers = np.array(centers)

    wX = values[:, 1:] * weights[:,None]

    for i in range(k_means_iterations):
       Euclid_distance = cdist(values[:, 1:], centers, 'sqeuclidean')
       min_indices = np.argmin(Euclid_distance, axis=1)

       for clusterID in range(k):
           if (clusterID in min_indices):
               cluster_idx = np.where(min_indices == clusterID)[0]
               numerator = np.sum(wX[cluster_idx], axis=0)
               denominator = np.sum(weights[cluster_idx])
               centers[clusterID, :] = 1/ denominator * numerator

    yield centers