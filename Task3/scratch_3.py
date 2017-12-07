import numpy as np
from scipy.spatial.distance import cdist

mFeats = 250

# Number of clusters
k = 200
# Initialise cluster centers
np.random.seed(23)
init_mu = np.random.randn(200, mFeats)

def getLabelIdx(Labelvec, label):
    label_idx = np.where(Labelvec == label)[0]

    return label_idx

def combineClusterElements(elements):
    counts = np.shape(elements)[0]
    sum_elements = np.sum(elements, axis = 0)

    return np.append(counts,sum_elements)

def sgd(X, init_mu = init_mu, maxIter = 10):

    t = 0
    mu_Mat = init_mu

    while t <= maxIter:
        t = t + 1
        Euclid_distance = cdist(X, mu_Mat)
        min_indices = np.argmin(Euclid_distance, axis=1)
        counts = np.zeros(min_indices.size)

        for sample in range(3000):
            min_index = min_indices[sample]
            counts[min_index] = counts[min_index] + 1
            rate = 1/counts[min_index]
            mu_Mat[min_index, :] = ((1 - rate ** 2) * mu_Mat[min_index, :]) + (rate ** 2 * X[sample, :])

    return np.column_stack([range(200),mu_Mat])


def mapper(key, value):

    mu_batch = sgd(value)

    yield 0, mu_batch  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    cluster_centers = []
    for i in range(200):
        label_idx = getLabelIdx(values[:,0],i)
        cluster_centers.append(np.sum(values[label_idx, 1:], axis = 0)/label_idx.size)
    #print(np.shape(cluster_centers))

    yield np.asarray(cluster_centers)
    #yield np.random.randn(200, 250)

###################

    centers = list()
    center_indices = np.random.choice(len(values), k)

    for idx in center_indices:
        centers.append(values[idx])

    centers = np.array(centers)

    for i in xrange(1, k_means_iterations + 1):
        Euclid_distance = cdist(values, centers, 'euclidean')

        min_indices = np.argmin(Euclid_distance, axis=1)
        counts = np.zeros(k)

        for sample in range(values.shape[0]):
            min_index = min_indices[sample]
            counts[min_index] = counts[min_index] + 1
            rate = 1 / counts[min_index]
            centers[min_index, :] = ((1 - rate ** 2) * centers[min_index, :]) + (rate ** 2 * values[sample, :])



###################################
##### Weighted ##################
centers = list()
    weights = values[:,0]
    center_indices = np.random.choice(len(values), k)

    for idx in center_indices:
        centers.append(values[idx,1:])

    centers = np.array(centers)
    print(np.shape(centers), np.shape(weights))

    for i in xrange(1, k_means_iterations + 1):
        Euclid_distance = cdist(values[:,1:], centers, 'euclidean')
        normalised_dist = Euclid_distance / Euclid_distance.sum(axis=1)[:, None]
        min_indices = np.argmin(normalised_dist, axis=1)
        min_norm_dist = np.amin(normalised_dist, axis=1)
        prod_dist_weights = np.multiply(min_norm_dist,weights )

        for clusterID in range(200):
            if(clusterID in min_indices):
                cluster_idx = np.where(min_indices == clusterID)[0]
                numerator = np.sum((values[cluster_idx, 1:].T * prod_dist_weights[cluster_idx]).T , axis = 0)
                denominator = np.sum(prod_dist_weights[cluster_idx])
                centers[clusterID, :] = np.divide(numerator, denominator)