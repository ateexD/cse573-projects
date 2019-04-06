# Plot utilities
import cv2
import matplotlib.pyplot as plt

def plot_key_points(img, keypoints):
    """
    Input - 
            img - Source image
            keypoints - Keypoints of above image
    Returns - None
    
    Plot image by convert to color.
    """
    image_to_plot = cv2.drawKeypoints(img, keypoints, np.array([]))
    convert_and_plot(image_to_plot)
    
def convert_and_plot(img):
    """
    Input - Source image
    Returns - None
    
    Plots images after converting.
    """
    plt.rcParams["figure.figsize"] = [20, 20]
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()