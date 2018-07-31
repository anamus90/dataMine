__author__ = "Sumana Srivatsa"
__email__ = "sumana.srivatsa@bsse.ethz.ch"

import numpy as np

SEED = 456
#print SEED
np.random.seed(SEED)

# Defining constants used for hashing and binning
LargestPrime = 21503 #2147483869
LSH_mod = 23
bands = 27
rows = 27
n = bands * rows
var = np.random.uniform(0, 1000,n)
cons = np.random.uniform(0, 1000,n)
var_lsh = np.random.uniform(0,100,rows)
cons_lsh = np.random.uniform(0,100,rows)


# Hashing function for min hash and lsh
def hash_func(x, a, b, const):
    return (a * x + b) % const


# Define the number of hash functions for min hashing. Default n = 10.
# Signature generated by min-hashing. Default length 10
def get_minHash_signatures(shingle, n = 10):

    signature = []
    # Looping through all the hash coefficients
    for j in range(n):

        minHashVal = LargestPrime
        # Looping to find the smallest hash value
        for i in shingle:
            val = hash_func(i, var[j], cons[j] , LargestPrime)

            if val < minHashVal:
                minHashVal = val

        signature.append(minHashVal)

    return signature


# Function to map the rows of the docID to the bucket
# Hashing function uses a bucket_prime number with default = 251
def map_sig_to_LSH_buckets(docID, signature, nbands = 5, nrows = 2):
    if len(signature) != nrows * nbands:
        raise 'Choice of bands and rows does not match length of signature! ' \
              'Choose bands and rows correctly. '

    buckets_array = []

    i = 0
    for band in range(nbands):
        current_band = signature[i:i + nrows]
        #bucket_prime = 251
        #key = int(sum(current_band) % bucket_prime)
        #key = int(sum(current_band))
        key = sum([(hash_func(current_band[j],var_lsh[j],cons_lsh[j],LSH_mod)) for j in range(len(current_band))])
        #print key
        buckets_array.append((str(key), docID))
        i = i + nrows

    return buckets_array

# Define mapper function
# Given an empty key and value (input string) the function returns all LSH keys for that page
def mapper( key, value):
    (docID, shingles) = value.split(None, 1)
    sig = get_minHash_signatures(map(int, shingles.split(' ')), n)
    for i in map_sig_to_LSH_buckets(docID,sig, bands, rows):
        yield i


# For a given key (bucket number) and values (set of all documents with the same hash key) return all possible pairs of similar documents
def reducer(key, values):
    pages = sorted(map(lambda x: int(x.split("_")[1]), values))
    for i in xrange(len(pages) - 1):
        for j in xrange(i + 1, len(pages)):
            if pages[i] < pages[j]:
                yield pages[i], pages[j]