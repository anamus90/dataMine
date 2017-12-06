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