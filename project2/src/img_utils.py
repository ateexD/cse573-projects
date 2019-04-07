import cv2
import functools
import numpy as np

from matcher import *

# Homography function
def get_homography(point_group_1, point_group_2, verbose=False):
    """
    Input - 
            point_group_[1 & 2] - 2 groups of points chosen randomly
            as a part of RANSAC
     
    Returns - Homography matrix
    """
    
    # Init P as None
    P = None
    
    # P matrix
    for point1, point2 in zip(point_group_1, point_group_2):
        # Get x, y & x', y'
        x, y = point1[0], point1[1]
        x_bar, y_bar = point2[0], point2[1] 
        
        point_matrix = [
            [x, y, 1, 0, 0, 0, -x * x_bar, -y * x_bar, -x_bar],
            [0, 0, 0, x, y, 1, -x * y_bar, -y * y_bar, -y_bar]
        ]
        # If this is the first pair of rows
        if P is None:
            P = np.array(point_matrix)
        # Or stack to exitsting
        else:
            P = np.vstack((P, point_matrix))
    
    # Get eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P.T.dot(P))
    
    # Get minimum eigenvalue's index
    min_eigenvalue_index = np.argmin(eigenvalues)
    
    # Sanity check - 1
    if verbose:
        print("Minimum eigenvalue found at ", min_eigenvalue_index, ":", eigenvalues[min_eigenvalue_index])
    
    # Get corresponding eigenvector
    min_eigenvector = eigenvectors[:, min_eigenvalue_index]
    
    # Normalize the eigenvector by L2 norm
    norm = np.linalg.norm(min_eigenvector)
    min_eigenvector /= norm
    
    # Sanity check - 2 
    # These values must be close to 0
    if verbose:
        print ("These values should be close to 0", np.round(P.dot(min_eigenvector), 2))
    
    # Homography matrix
    H = min_eigenvector.reshape(3, 3)
    
    # Sanity check - 3
    if verbose:
        for i, point in enumerate(point_group_1):
            point = [point[0], point[1], 1]
            point_2_predicted = np.dot(H, point)
            point_2_predicted = (point_2_predicted / point_2_predicted[-1])[:2]
            print("Original point:", point[:-1])
            print("Predicted point:", point_2_predicted)
            print("Actual point:", point_group_2[i])
            print()
    
    return H

def ransac(matches, kp1, kp2, epsilon=5.0, batch_size=4, max_iter=10, verbose=False):
    """
    Input - Matches from one_NN_matcher(..)
    Returns - Inlier homography matrix
    """    
    
    # Store best homography 
    best_H = None
    best_matches = []
    best_inilier_count = -1
        
    for curr_iter in range(max_iter):
        if verbose and (curr_iter + 1) % 5 == 0:
            print("Currently at iter #", (curr_iter +  1), sep="")
            
        # Randomly permute matches list
        np.random.shuffle(matches)
        
        # Iter through remaining matches
        for i in range(0, len(matches), batch_size):
            
            # Default termination condition
            if i + batch_size >= len(matches):
                break

            # Current random subsample
            point_group_1, point_group_2 = [], []
            for match in matches[i : i + batch_size]:
                point_group_1.append(kp1[match.queryIdx].pt)
                point_group_2.append(kp2[match.trainIdx].pt)

            # Compute Homography
            H = get_homography(point_group_1, point_group_2)

            # Initialize inlier Count
            inlier_count = 0

            # Get matches of current model
            curr_model_matches = []
            
            # Add inliers; remove outliers
            for m in matches:
                # Get left point
                point_1 = [kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1], 1]
                
                # Get actual right point
                point_2_actual = [kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]]

                # Predict right point
                point_2_predicted = np.dot(H, point_1)
                point_2_predicted = (point_2_predicted / point_2_predicted[-1])
                point_2_predicted = point_2_predicted[:-1]
                
                # Compute the L2 distances between predicted & actual point 
                diff = np.linalg.norm(np.subtract(point_2_actual, point_2_predicted))

                # Increment inlier_count if necessary
                if diff < epsilon:
                    inlier_count += 1
                    curr_model_matches.append(m)

            # Update best model if necessary
            if inlier_count > best_inilier_count:
                best_inilier_count = inlier_count
                best_H = H
                best_matches = curr_model_matches
                
                # Terminate early based on count
                if best_inilier_count > int(0.3 * len(kp1)):
                    if verbose:
                        print("-------------------\nCutting short early\n-------------------")
                        # Print stats and return best_H
                        print("Inliers -", best_inilier_count)
                        print("Outliers -", len(kp1) - best_inilier_count)
                    return best_H, best_matches
                    
    # Terminate normally with best_H seen
    if verbose:
        print("Inliers -", best_inilier_count)
        print("Outliers -", len(kp1) - best_inilier_count)
    return best_H, best_matches


def get_direction(kp1, kp2, des1, des2, img1, img2, verbose=True):
    """
    Input - 
            kp[1 & 2] - Keypoints from images 1 & 2
            des[1 & 2] - Descriptors from images 1 & 2
            img[1 & 2] - Imgaes 1 & 2
            
    Returns - Left & Right images
    """
    _, w, _ = img2.shape
    
    def get_distances(kp1, kp2, des1, des2):
        """
        Utility function to get left & right distances
        """
        matches = get_good_matches(kNN_matcher(des1, des2))
        _, matches = ransac(matches, kp1, kp2, max_iter=10)

        matches = sorted(matches, key=lambda x: x.distance)
        distances = []
        
        for match in matches:
            point1, point2 = kp1[match.queryIdx].pt, kp2[match.trainIdx].pt
            point1, point2 = np.array(point1), np.array(point2)

            point2[0] += w

            distance = np.linalg.norm(point1 - point2)
            distances.append(distance)
            
        return distances
        
    left_distances = get_distances(kp1, kp2, des1, des2)
    right_distances = get_distances(kp2, kp1, des2, des1)
    
    if np.mean(left_distances) <= np.mean(right_distances):
        return -1
    return 1