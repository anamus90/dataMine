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

def mapper(key, value):

    cluster_value = []
    Euclid_distance = cdist(value, init_mu)
    min_index = np.argmin(Euclid_distance, axis = 1)

    for i in range(1,201):
        if i in min_index:
            label_idx = getLabelIdx(min_index,i)
            cluster_value.append(np.append(i,combineClusterElements(value[label_idx, :])))
        else:
            cluster_value.append(np.append(i, np.append(1, init_mu[i-1, :])))

    yield 0, cluster_value  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    cluster_centers = []
    for i in range(1,201):
        label_idx = getLabelIdx(values[:,0],i)
        sum_array = combineClusterElements(values[label_idx, 1:(mFeats+2)])
        #print(np.shape(sum_array))
        cluster_centers.append(sum_array[2:(mFeats+2)]/sum_array[1])
        #print(sum_array[1])

    yield np.asarray(cluster_centers)
    #yield np.random.randn(200, 250)
