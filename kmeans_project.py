import librosa
import numpy as np
import soundfile as sf
import random


def separateMelReturnJth(mel, melc, j, nmfcc):
    """
    Converts the mfcc matrix into a matrix where only points from cluster j are included
    :param mel:     the mfcc matrix
    :param melc:    the classification of points into clusters
    :param j:       cluster number j
    :param nmfcc:   the number of
    :return:        a matrix where only points in cluster j are included, silence elsewhere
    """
    # Initialises the cluster matrix as a zero matrix
    nmel = len(mel)
    vect = np.zeros((nmel, nmfcc))

    # Constructs a silent vector. This value corresponds to silence
    silence = np.zeros(nmfcc)
    silence[0] = -1131.3708499

    for i in range(nmel):
        # Checks if point i is in cluster j. If not, the point is set to be silent
        if melc[i] == j:
            vect[i] = mel[i]
        else:
            vect[i] = silence
    return vect


def writeSeparateFiles(file, nClusters, nMfcc, mel, melc, **kwargs):
    """
    Writes a file for each cluster with only the audio from that cluster
    :param file:        the original file
    :param nClusters:   number of clusters
    :param nMfcc:       number of features in the mfcc matrix
    :param mel:         the mfcc matrix
    :param melc:        the classification of all the points
    :return:
    """
    for i in range(nClusters):
        # separates so that only points from cluster i gets included
        vect = separateMelReturnJth(mel, melc, i, nMfcc)
        # Transposes so the file is in correct format for the inverse operations
        vec = np.transpose(vect)

        # converts the cluster matrix into an audio time series
        vec_spec = librosa.feature.inverse.mfcc_to_mel(vec, n_mels=nMfcc)
        vec_audio = librosa.feature.inverse.mel_to_audio(vec_spec)

        # writes the audio time series to a file, labeled by cluster and the number of features
        sf.write("C" + str(i) + "F" + str(nMfcc) + file, vec_audio, 22050)


def assignPoints(matrix, centroidsAtm, nRows, k):
    """
    Assigns points to their correct current cluster
    :param matrix:  nRow points to be clustered with nCol attributes
    :param centroidsAtm:    the centroids to the current clusters
    :param nRows:   number of points
    :param k:       number of clusters
    :return:        returns an array with classification of each point to a cluster
    """
    # initialises the classification
    classification = np.zeros(nRows)

    for i in range(nRows):
        # Selects point i. Computes the distance to the first centroid, to compare with the other distances
        point = matrix[i]
        centroid = centroidsAtm[0]
        minDist = np.linalg.norm(point - centroid)

        for j in range(1, k):
            # Compute the distance to centroid j
            centroid = centroidsAtm[j]
            testDist = np.linalg.norm(point - centroid)

            # Compares the current distance to the minimum distance
            if testDist < minDist:
                # Sets the current distance to the new minimum, and classifies point i as belonging to cluster j
                minDist = testDist
                classification[i] = j

    return classification


def costFunction(matrix, classification, centroids):
    """
    Computes the cost of the clustering
    :param matrix: points to be clustered
    :param classification: the classification of each point
    :param centroids: the centroids of the clusters
    :return: the cost of the clustering
    """
    # Initialises the cost
    cost = 0

    for i in range(len(centroids)):
        for j in range(len(classification)):
            # Adds the distance between point j and centroid i if point j belongs to cluster i
            if classification[j] == i:
                cost += np.linalg.norm(matrix[j] - centroids[i]) ** 2

    return cost


def farthestPoints(points, centroids, nPoints):
    """
    Computes the point which is the farthest from the centroids
    :param points:  candidates to the next centroid
    :param centroids: the points already selected as centroids
    :param nPoints: the number of points
    :return: a new centroid and the index of that centroid in points
    """
    # initialises the distance and the centroid
    maxDist = 0
    newCentroid = points[0]

    for i in range(nPoints):
        # initialises current distance
        distance = 0

        # adds the distance to every centroid from point i
        for centroid in centroids:
            distance += np.linalg.norm(np.array(points[i]) - np.array(centroid))

        # sets point i to be the new centroid if the distance exceeds the previous maximum distance
        if distance >= maxDist:
            maxDist = distance
            newCentroid = points[i]
            index = i

    return newCentroid, index


def farthestFirst(matrix, k, nRows):
    """
    Computes centroids using the farthest first heuristic.
    :param matrix: the points
    :param k:   the number of clusters
    :param nRows: the number of points
    :return:    returns a starting position for the centroids
    """
    # Picks a random point to be the first centroid
    centroids = []
    rand = random.randint(0, nRows - 1)
    centroids.append(matrix[rand])

    # converts the points to list format and removes the first centroid
    copyMatrix = matrix.tolist()
    copyMatrix.pop(rand)

    for i in range(1, k):
        # Finds the point with largest distance from the centroids. Adds this to centroids.
        # Removes this point from the copymatrix
        newCentroid, index = farthestPoints(copyMatrix, centroids, len(copyMatrix))
        centroids.append(newCentroid)
        copyMatrix.pop(index)

    return np.array(centroids)


def moveCentroids(centroidsAtm, classification, matrix, k, nRows):
    """
    Move the centroids to the middle of the current cluster
    :param centroidsAtm:    the current centroids
    :param classification:  classification of the points in the clusters
    :param matrix:          the points
    :param k:               number of clusters
    :param nRows:           number of points
    :return:                the moved centroids
    """
    for i in range(k):
        # nClus is the number of points in the given cluster
        nClus = 0
        # Initialises the sum of points iin a cluster to 0
        sumClus = np.zeros(len(centroidsAtm[0]))

        for j in range(nRows):
            # Adds point j to the sum if it is in cluster i. Increments the number of points in the cluster
            if classification[j] == i:
                nClus += 1
                sumClus = np.add(sumClus, matrix[j])

        # Adds the scaled sum as a new centroid
        sumClus = sumClus * (1 / nClus)
        centroidsAtm[i] = sumClus
    return centroidsAtm


def centroidsConverge(centroidsPrev, centroidsAtm):
    """
    Checks if the centroids from the previous iteration is equal to the current iteration
    :param centroidsPrev: Previous centroids
    :param centroidsAtm:  Current centroids
    :return:              returns a boolean
    """
    return (centroidsPrev == centroidsAtm).all()


def KmeansIteration(matrix, k):
    """
    Computes Kmeans
    :param matrix:  the points to be clustered
    :param k:       the number of clusters
    :return:        returns a classification of the points and the centroids at the end
    """
    # Computes the number of points and number of features
    nRows = len(matrix)
    nCol = len(matrix[0])

    # Initialises the current centroids, and the previous
    centroidsPrev = np.zeros((k, nCol))
    centroidsAtm = farthestFirst(matrix, k, nRows)

    # checks if the centroids converge
    while not centroidsConverge(centroidsPrev, centroidsAtm):
        # computes a classification of the points
        classification = assignPoints(matrix, centroidsAtm, nRows, k)

        # sets the previous centroids to the current centroids, and computes the new cluster centroids
        centroidsPrev = centroidsAtm
        centroidsAtm = moveCentroids(centroidsAtm, classification, matrix, k, nRows)

    # computes a final classification
    classification = assignPoints(matrix, centroidsAtm, nRows, k)
    return classification, centroidsAtm


def Kmeans(matrix, k, iterations):
    """
    Runs the Kmeans algorithm iterations number of times and chooses the classification with the lowest cost
    :param matrix:      the data which will be clustered. The data comprises of nRows datapoints with nCols features
    :param k:           the number of clusters
    :param iterations:  the number of trials of the k-means algorithm. The version with the least cost assosiated is selected
    :return:            returns an array of length nRows, where each datapoint is assigned to a cluster and the cost function for the best of the iterations
    """
    # runs Kmeans once, to compare every subsequent run to
    classification, centroids = KmeansIteration(matrix, k)
    smallCost = costFunction(matrix, classification, centroids)
    trueClass = classification

    for i in range(1, iterations):
        # Runs Kmeans and calculates the cost
        classification, centroids = KmeansIteration(matrix, k)
        cost = costFunction(matrix, classification, centroids)

        # Checks if the cost of the new iteration is less than the previous. If so, assigns new values
        if cost <= smallCost:
            smallCost = cost
            trueClass = classification

    return trueClass, smallCost


def mfccKmeans(file, nClusters, nMfcc, inits):
    """
    Separates an audio file by using mfcc representation and Kmeans clustering
    :param file:        the file to separate
    :param nClusters:   the number of clusters
    :param nMfcc:       the number of features in the mfcc matrix
    :param inits:       the number of times to run Kmeans with a new initialisation
    :return: the classification of the points and the cost of our clustering
    """

    # Gets an audio time series from the file and the sample rate
    sound, sr = librosa.load(file)

    # Transforms the audio time series into a mfcc matrix with nMfcc features
    mel = librosa.feature.mfcc(sound, n_mfcc=nMfcc, n_mels=nMfcc)

    # Transposes the mfcc matrix so that Kmeans classifies correctly
    newMel = np.transpose(mel)

    # Runs Kmeans on the mfcc matrix
    classification, cost = Kmeans(newMel, nClusters, inits)

    # Writes nClusters different files based on our clustering
    writeSeparateFiles(file, nClusters, nMfcc, newMel, classification)

    return classification, cost
