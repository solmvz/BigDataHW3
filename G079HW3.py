# Import Packages
from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
import math


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MAIN PROGRAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def main():
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Initialize variables
    filename = sys.argv[1]
    k = int(sys.argv[2])
    z = int(sys.argv[3])
    L = int(sys.argv[4])
    start = 0
    end = 0

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # Read points from file
    start = time.time()
    inputPoints = sc.textFile(filename, L).map(lambda x: strToVector(x)).repartition(L).cache()
    N = inputPoints.count()
    end = time.time()

    # Pring input parameters
    print("File : " + filename)
    print("Number of points N = ", N)
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Number of partitions L = ", L)
    print("Time to read from file: ", str((end - start) * 1000), " ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)  # Solving ... Where to use L

    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()
    print("Objective function = ", objective)
    print("Time to compute objective function: ", str((end - start) * 1000), " ms")


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method strToVector: input reading
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def strToVector(str):
    out = tuple(map(float, str.split(',')))
    return out


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method squaredEuclidean: squared euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def squaredEuclidean(point1, point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i] - point2[i])
        res += diff * diff
    return res


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method euclidean:  euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def euclidean(point1, point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i] - point2[i])
        res += diff * diff
    return math.sqrt(res)


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points, k, z, L):
    # ------------- ROUND 1 ---------------------------
    start = time.time()
    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k + z + 1))
    end = time.time()
    print("Time to compute Round 1: ", str((end - start) * 1000), " ms")
    # END OF ROUND 1

    # ------------- ROUND 2 ---------------------------

    start = time.time()
    elems = coreset.collect()
    coresetPoints = list()
    coresetWeights = list()
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])

    # ****** ADD YOUR CODE
    # ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
    # ****** Measure and print times taken by Round 1 and Round 2, separately
    # ****** Return the final solution
    solution = SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, 0)
    end = time.time()
    print("Time to compute Round 2: ", str((end - start) * 1000), " ms")
    return solution


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter, points):
    partition = list(iter)
    centers = kCenterFFT(partition, points)
    weights = computeWeights(partition, centers)
    c_w = list()
    for i in range(0, len(centers)):
        entry = (centers[i], weights[i])
        c_w.append(entry)
    # return weighted coreset
    return c_w


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points, k):
    idx_rnd = random.randint(0, len(points) - 1)
    centers = [points[idx_rnd]]
    related_center_idx = [idx_rnd for i in range(len(points))]
    dist_near_center = [squaredEuclidean(points[i], centers[0]) for i in range(len(points))]

    for i in range(k - 1):
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0]  # argmax operation
        centers.append(points[new_center_idx])
        for j in range(len(points)):
            if j != new_center_idx:
                dist = squaredEuclidean(points[j], centers[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
                    related_center_idx[j] = new_center_idx
            else:
                dist_near_center[j] = 0
                related_center_idx[j] = new_center_idx
    return centers


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points, centers):
    weights = np.zeros(len(centers))
    for point in points:
        mycenter = 0
        mindist = squaredEuclidean(point, centers[0])
        for i in range(1, len(centers)):
            dist = squaredEuclidean(point, centers[i])
            if dist < mindist:
                mindist = dist
                mycenter = i
        weights[mycenter] = weights[mycenter] + 1
    return weights


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def SeqWeightedOutliers(P, w, k, z, alpha):
    #
    # ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
    #
    weights = np.asarray(w)
    n = len(P)
    g = 1
    r = sys.maxsize
    distances = np.zeros((n, n), dtype="float64")

    # (a) Compute distance between each point
    for i in range(n):
        for j in range(n):
            if i < j:
                distances[i, j] = euclidean(P[i], P[j])
            elif j < i:
                distances[i, j] = distances[j, i]

    # (b) r = (min-distance(r) between first k+z+1 points)/2
    for i in range(k + z + 1):
        for j in range(k + z + 1):
            if i < j:
                if distances[i, j] < r:
                    r = distances[i, j]
    # Should r be
    r /= 2

    # =====================================================
    # More Optimize Code that uses scipy (TO use uncomment this one  and comment (a) and (b))
    # distances=distance.cdist(P[:k+z+1], P[:k+z+1], 'euclidean')
    # print(distances)

    # r=np.min(distances[(np.ones((len(distances),len(distances)))-np.eye(len(distances))).astype(bool)])/2
    # ==========================================================

    print("Initial guess = ", r)
    while (True):
        Z = P.copy()
        solution = []
        Wz = sum(weights)
        while (len(solution) < k and Wz > 0):
            max = 0
            for x in range(n):
                ball_weight = 0
                for j in range(len(Z)):  # ball_weight= Sum(y in Bz(x,(1+2a)r W(y))
                    if Z[j] != -1:
                        if distances[x, j] <= (1 + 2 * alpha) * r:
                            ball_weight += weights[j]

                if ball_weight > max:
                    max = ball_weight
                    newcenter = x

            solution.append(P[newcenter])

            for y in range(len(Z)):  # look for points in Z
                if Z[y] != -1:
                    if distances[newcenter, y] <= (3 + 4 * alpha) * r:
                        Wz -= weights[y]
                        Z[y] = -1

        # plot_graph(np.ndarray(P),S,Z) #Code for Plot Library
        if Wz <= z:
            print("Final guess = ", r)
            print("Number of guesses = ", g)
            # print(S)
            return solution
        else:
            r = 2 * r
            g += 1


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective(P, solution, z):
    #To make this 2 round map reduce version of computeObjective we first divide in l partition (P is already Partitioned at before the methods call)
    #First round, for each partition of points we find the z+1 top distances from point in the partition to the closest center
    #We merge all those L*(Z+1) distances and we select the Top (Z+1). Now the minimun distances out of these will be the
    #objective function values
    return min(
        P.mapPartitions(lambda x: computeObjectiveReduce(x, solution, z))  #Round 1 Reduce
            .top(z + 1))                                                   #Round 2 Reduce


def computeObjectiveReduce(P, solution, z):
    min_dist = []
    for x in P:
        min = sys.maxsize
        for s in solution:
            dxs = euclidean(x, s)
            if dxs < min:
                min = dxs
        min_dist.append(min)
    min_dist.sort(reverse=True)

    return min_dist[0:z]


# Just start the main program
if __name__ == "__main__":
    main()

