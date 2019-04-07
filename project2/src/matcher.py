import cv2
import numpy as np

# Matcher - based on descriptors
def kNN_matcher(des1, des2, k=2):
    """
    Input - 
            des1 & des2 - Descriptor matrices for 2 images
            k - Number of nearest neighbors to consider
    Returns - A vector of nearest neighbors of des1 & their indices for keypoints
    
    Mnemonic - des1 is like xtest, des2 is like xtrain
    """

    # Compute the L2 equations
    distances = np.sum(des1 ** 2, axis=1, keepdims=True) + np.sum(des2 ** 2, axis=1) - 2 * des1.dot(des2.T)
    distances = np.sqrt(distances)
    
    
    # Get smallest indices 
    min_indices = np.argsort(distances, axis=1)
    
    # Init ndarray 
    nearest_neighbors = []
    
    # Iter for nearest neighbors
    for i in range(min_indices.shape[0]):
        neighbors = min_indices[i][:k]
        curr_matches = []
        for j in range(len(neighbors)):
            match = cv2.DMatch(i, neighbors[j], 0, distances[i][neighbors[j]] * 1.)
            curr_matches.append(match)
        nearest_neighbors.append(curr_matches)
    
    return nearest_neighbors

def get_good_matches(matches, ratio_test_param=0.75):
    """
    Input - 
            matches - n x k matches from kNN_matcher
            ratio_test_param - Lowe's ratio test parameter
    """
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_test_param * n.distance:
            good_matches.append(m)
    return good_matches