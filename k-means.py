'''
Tommy Yu
Jan-22-2018
Revised: Jan-30-2018
Written in Python 3
'''

import numpy as np
import pandas as pd
import random
import multiprocessing as mp
from functools import partial
import timeit
import sys

df = pd.read_csv("County_Mortgage_Funding.csv")
df = df.rename(columns={'Unnamed: 0': 'ID'})
random.seed(12345)

# For every entry, create a list containing its entry_id, array of all attributes, and placeholder for No. of centroid
records = [[int(row[0]), row[1:], 0] for row in df.values]


def find_nearest_centroid(record, centroids):
    '''
    Find the nearest centroid based on Eculidean distance,
    update to reflect the No. of the centroid.
    When passed into multiprocessing, this only yields a copy of the updated record,
    since the records in the parent process cannot be affected.
    '''
    dist_min = float("inf")
    for i, centroid in enumerate(centroids):
        dist = np.linalg.norm(record[1] - centroid)
        if dist < dist_min:
            dist_min = dist
            clus_n = i + 1
    record[2] = clus_n
    return record


def update_centroid(clusters):
    '''
    Given the clusters after one pass,
    calculate the new centroids.
    '''
    new_centroids=[]
    for cluster in clusters:
        new_centroids.append(np.mean(cluster, axis=0))
    return new_centroids


def finalized(centroids, new_centroids):
    '''
    Determine whether centroids have moved after another pass.
    '''
    return [tuple(c) for c in centroids] == [tuple(c) for c in new_centroids]


def print_result(centroids, results):
    '''
    Display attributes of all centroids and assignments of all entries.
    '''
    print("Assignments of entries to centroids:")
    for r in results:
        print("Entry "+str(r[0])+" is assigned to centroid No. " + str(r[2]))
    print("")
    print("Cluster centroids - values in tuple corresponds to:", tuple(df.columns[1:]))
    for i, centroid in enumerate(centroids):
        print("No."+str(i+1)+" centroid: ", tuple(centroid))


def run_k_means(k, cores=mp.cpu_count(), verbose=True):
    '''
    K-means algorithm.
    '''
    # Randomly select k entries as initial centroids.
    centroids = [record[1] for record in random.sample(records, k)]
    
    pool = mp.Pool(cores)

    while True:

    ######## partial must be within the while loop #########
        find_nearest_centroid_ = partial(find_nearest_centroid, centroids=centroids)

        # Initiate empty clusters
        # Fill the clusters (in the parent process) based on results from child processes
        # Then calculate new centroids
        clusters = [[] for i in range(k)]
        results=pool.map_async(find_nearest_centroid_, records).get()
        for r in results:
            clusters[r[2]-1].append(r[1])
        new_centroids = update_centroid(clusters)
        
        # If at least centroid shifted, go through the process again, until all centroids are finalized
        if not finalized(centroids, new_centroids):
            centroids = new_centroids
        else:
            if verbose:
                print_result(centroids, results)
            break
    pool.close()
    pool.join()

if __name__ == "__main__":
    k = int(input("What is k?\n"))
    run_k_means(k)

    test = input("Test time (average of 5 executions)? y/n\n")
    if test == "y" or test == "Y":
        def t1():
            run_k_means(k, 1, False)
        def t2():
            run_k_means(k, 2, False)
        def t3():
            run_k_means(k, 3, False)
        def t4():
            run_k_means(k, 4, False)

        print("Using 1 core takes {} s".format(timeit.timeit(t1, number=5)/5))
        print("Using 2 cores takes {} s".format(timeit.timeit(t2, number=5)/5))
        print("Using 3 cores takes {} s".format(timeit.timeit(t3, number=5)/5))
        print("Using 4 cores takes {} s".format(timeit.timeit(t4, number=5)/5))

    else:
        sys.exit()