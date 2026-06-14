import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


#
# NO MORE MODULES ALLOWED
#

# Gaussian filter = Blurring + Smoothing
# It makes the image smoother by averaging nearby pixels.
def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    ax = np.arange(-(ksize // 2), ksize // 2 + 1)  #np.arange() = "array range" — creates an array of numbers in sequence.
    # We create coordinates centered around zero. so the Gaussian peak is at the center of the kernel array, not at a corner.
    xx, yy = np.meshgrid(ax, ax) #It takes your 1D line and makes a 2D grid.
    #xx (horizontal distances): yy (Vertical distances)
    #The Gaussian formula calculates distance from center: so we need both x AND y values at every position in the kernel to compute the distance.
    # why exp: Converts distance into weight → Far pixels get small weights, near pixels get large weights an
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)) #Gaussian = e^(-(distance² from center) / (2σ²))  (2σ²)-> controls how wide or narrow the blur is?
    #The distance tells us "how far each pixel is from the center
    # ---for minus: It makes the Gaussian peak at the center and fall off to the sides
    kernel = kernel / kernel.sum() #Normalize the kernel. If it's not 1.0, you're accidentally brightening or darkening the entire image! 🎯
    filtered = convolve(img_in, kernel).astype(int)
    #The Gaussian blur smooths the image by averaging nearby pixels. So values change! That's the whole point of blurring. ✨
    return kernel, filtered


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """

    #Sobel detects brightness changes in two directions:
    #gx: Changes from LEFT to RIGHT (vertical edges)
    #gy: Changes from TOP to BOTTOM (horizontal edges)

    #A 3×3 matrix (kernel) that detects LEFT-RIGHT brightness changes.
    #- Subtracts: RIGHT - LEFT = brightness change
    sobel_x = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=float)
    #A 3×3 matrix that detects TOP-BOTTOM brightness changes.
    #Subtracts: BOTTOM - TOP = brightness change
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=float)
    #Apply Sobel X kernel
    #np.flip(sobel_x): Flips the kernel. True convolution
    # Matters for Sobel but not for Gaussian blur For edge detection,
    # the direction matters. Flipping ensures the kernel detects edges in the correct direction. Without the flip, left and right would be reversed
    gx = convolve(img_in, np.flip(sobel_x)).astype(int)
    gy = convolve(img_in, np.flip(sobel_y)).astype(int)
    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    #g (magnitude): How strong is the edge? (edge strength)
    #theta (direction): Which way does the edge point? (edge angle)
    #  Pythagorean theorem if needed say it not no need
    g = np.sqrt(gx ** 2 + gy ** 2).astype(int) ## square to get positive value as we are checking how strong the edge is and squreroot it back.
    #arc tangent of two values" — calculates the angle.
    theta = np.arctan2(gy, gx) #direction gives angle in radians arctan(computes the direction of edge). 0 degree is edge vertical check right or left or vice versa
    #in non-maximum suppression, we need to know: Which direction should we check neighbors on?"
    #returns radians
    return g, theta


#Converting continuous angles (any angle like 23.5°, 91.2°) into 4 discrete angles (0°, 45°, 90°, 135°).
def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    #np.degrees(angle) converts the radians to degree
    angle_deg = np.degrees(angle) % 180 #% 180 reduces angles from 0-360° to 0-180° because edge directions repeat every 180° (opposite directions look identical)."
    #180° / 4 = 45° each 22.5° (45/2) 67.5° (45 + 22.5) 112.5° (90 + 22.5) 57.5° (135 + 22.5) Each boundary is halfway between two main directions.
    if (angle_deg >= 0 and angle_deg < 22.5) or (angle_deg >= 157.5 and angle_deg < 180):
        return 0
    elif angle_deg >= 22.5 and angle_deg < 67.5:
        return 45
    elif angle_deg >= 67.5 and angle_deg < 112.5:
        return 90
    else:
        return 135

# Thins thick edges to 1 pixel wide by keeping only the local maximum in the gradient direction.
def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    rows, cols = g.shape  #gets rows and columns
    max_sup = np.zeros((rows, cols), dtype=float) #creates a new array of zeros with same rows and columns as g(image)
    #A kernel needs neighbors on all sides
    #XX why no padding: Only authentic pixels with real neighbors are compared
    for i in range(1, rows - 1):  # Skip row 0 and last row Top-left pixel (X) has no left neighbor, no top neighbor!
        for j in range(1, cols - 1):  # Skip column 0 and last column
            angle = convertAngle(theta[i, j])
            if angle == 0:
                neighbors = (g[i, j - 1], g[i, j + 1])  # left and right
            elif angle == 45:
                neighbors = (g[i - 1, j + 1], g[i + 1, j - 1])  # top-right  bottom-left
            elif angle == 90:
                neighbors = (g[i - 1, j], g[i + 1, j])   #top and bottom
            else:
                neighbors = (g[i - 1, j - 1], g[i + 1, j + 1])  #top-left  bottom-right

    #Is current pixel ≥ neighbor 1?  AND Is current pixel ≥ neighbor 2? If BOTH are true → keep it If ANY is false → delete it
            if g[i, j] >= neighbors[0] and g[i, j] >= neighbors[1]:
                max_sup[i, j] = g[i, j]

    return max_sup

#Converts the thin edges into final binary edges using two thresholds instead of one.
def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    rows, cols = max_sup.shape
    result = np.zeros((rows, cols), dtype=np.uint8)

    strong = max_sup >= t_high   # Creates a true/false map of pixels ≥ high threshold. boolean array
    weak = (max_sup >= t_low) & (max_sup < t_high)  # Creates a true/false map of pixels between low and high threshold.

    result[strong] = 255 # Sets all pixels where strong == True to white (255).

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak[i, j]: #check if the pixle is in the middle or not
                neighborhood = strong[i - 1:i + 2, j - 1:j + 2]  #all 9 values (including center)
                if neighborhood.any(): #"Is ANY value True in this 3×3 box?"
                    result[i, j] = 255

    return result


def canny(img):
    kernel, gauss = gaussFilter(img, 5, 2)

    gx, gy = sobel(gauss)

    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    g, theta = gradientAndDirection(gx, gy)

    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    maxS_img = maxSuppress(g, theta)

    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result


# Main execution
# if __name__ == "__main__":
#     # Create a synthetic test image with clear edges
#     img = np.zeros((200, 200), dtype=float)
#     img[50:150, 50:150] = 200  # White square
#     img[75:125, 75:125] = 50  # Dark square inside
#
#     print(f"Image shape: {img.shape}")  # Should be (200, 200)
#     print(f"Image dtype: {img.dtype}")
#
#     # Run Canny edge detection
#     edges = canny(img)
#
#     # Display final result
#     plt.figure(figsize=(8, 6))
#     plt.imshow(edges, 'gray')
#     plt.title('Canny Edge Detection - Final Result')
#     plt.colorbar()
#     plt.show()

if __name__ == "__main__":
    # Load a real image (e.g., from the exercise folder)
    img = plt.imread("data/contrast.jpg")

    edges = canny(img)

    plt.imshow(edges, 'gray')
    plt.title('Canny Edge Detection - Real Image')
    plt.show()