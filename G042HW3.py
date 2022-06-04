# Import Packages
from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
import math
from pyspark import SparkContext, SparkConf
import random as rand
import psutil
import os
import sys
from pyspark.sql import SparkSession

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MAIN PROGRAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def main():
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Spark setup
    #conf = SparkConf().setAppName('HomeWork3').setMaster("local[*]")
    #sc = SparkContext(conf=conf)

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
    solution = MR_kCenterOutliers(inputPoints, k, z, L)

    # Compute the value of the objective function
    print(solution)
    start = time.time()
    objective = computeObjective(inputPoints, solution, z, L)
    end = time.time()
    print("Objective function = ", objective)
    print("Time to compute objective function: ", str((end - start) * 1000), " ms")


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
def minDistance(P, n):
    subset = np.array(P[:n])
    min_d = float('inf')
    while len(subset):
        i = subset[0]
        subset=np.delete(subset,0,0)
        for j in subset:
            #current_d = euclidean(i, j)
            current_d = np.sqrt(np.sum(np.square(i - j)))

            if  current_d < min_d:
                min_d = current_d
    # print(min_d)
    return min_d / 2

def weightInRadius(point_array, weight_array, x, x_w, op):
    # we used the np to make this step more efficient
    # we firstly compute the euclidean distances from x to all the point in point_array
    euclidean_distance = np.sqrt(np.sum(np.square(point_array - x), 1))

    # we find the indeces of the point that are inside the first ball
    indeces = np.where(euclidean_distance < op)

    #we get the weight of those points and we remove
    #the weight of x
    return weight_array[indeces].sum() - x_w
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

    coreset = points.mapPartitions(lambda iterator: extractCoreset(iterator, k + z + 1))

    # END OF ROUND 1

    # ------------- ROUND 2 ---------------------------

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
    return SeqWeightedOutliers(coresetPoints, coresetWeights, k, z, 2)[0]




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
def SeqWeightedOutliers(inputPoints, weights, k, z, alpha=0):
    #we compute the first circle base radious
    r = minDistance(inputPoints, k + z + 1)
    r_init = r;
    num_iter = 1
    while True:
        # To make the this function more efficient we are going to use Numpy arrays
        # to reppresent Z.
        # Z_weight is going to store the weights of the point in Z, using same indexing
        Z = np.zeros((len(inputPoints), len(inputPoints[0])))
        Z_weight = np.zeros(len(inputPoints))
        for index in range(len(inputPoints)):
            Z[index] = np.asarray(inputPoints[index])
            Z_weight[index] = weights[index]

        # We initialize the set of the centers solutions
        S = []
        # We compute the initial Weight of Z
        W_z = np.sum(Z_weight)

        op = (1 + 2 * alpha) * r
        while len(S) < k and W_z > 0:
            #we initialize max distance and new_center
            MAX = -1
            new_center = None

            #for each point we compute the weight inside the relative ball
            for index in range(len(Z)):

                x = Z[index]
                x_w = Z_weight[index]

                ball_weight = weightInRadius(Z, Z_weight, x, x_w, op)
                if ball_weight > MAX:
                    MAX = ball_weight
                    new_center = x

            #we add the new center to the solutions
            S.append(tuple(new_center))

            #we collect the index of the point outside the sedond ball of the new centers
            points_to_maintain = []
            for indeces in range(len(Z)):
                if np.sqrt(np.sum(np.square(Z[indeces] - new_center))) <= ((3 + 4 * alpha) * r):
                    W_z -= Z_weight[indeces]
                else:
                    points_to_maintain.append(indeces)

            # remove points that are not in the bigger circle from Z;
            Z = Z[points_to_maintain]
            Z_weight = Z_weight[points_to_maintain]

        if W_z <= z:
            break
        else:
            r = 2 * r
            num_iter += 1

    return S, r_init, r, num_iter

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective(P, S, z, l):
    return min(P.repartition(numPartitions=l)
               .mapPartitions(lambda Pi: computeObjectiveAux(Pi, S, z+1))
               .top(z+1))

def computeObjectiveAux(P, S, n):
    # At first we compute for each point the closest center.
    # for each point we save:
    # - the distance to the closest center
    # - the center
    distances = []
    for point in P:
        min_distance = float('inf')
        closest_center = None
        for center in S:
            distance = euclidean(point, center)
            if min_distance > distance:
                min_distance = distance
                closest_center = center
        distances.append(min_distance)

    # We sort the list on the distances
    distances = sorted(distances, reverse=True)

    return [dist for dist in distances[0:n]]


#
# ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
#


# Just start the main program
if __name__ == "__main__":
    main()