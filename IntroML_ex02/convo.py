import ks
from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    ks = np.zeros((ksize, ksize))  #Create an empty grid of zeros. 2d matrix
    half = ksize // 2 # Find the center.
    for i in range(ksize):
        for j in range(ksize):
            #For each position, calculate how far it is from the center. because the wight be high near center and low from from center
            x = i - half    # Distance from center (left-right)
            y = j - half    # Distance from center (up-down)
            #why exp: Converts distance into weight → Far pixels get small weights, near pixels get large weights and we can get smooth bell curve
            ks[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            #NORMALIZE - Make sure all values add up to 1. This is important so the image brightness doesn't change!
    ks = ks / ks.sum()
    return ks


def slow_convolve(arr, k):
    k_flipped = k[::-1, ::-1]  # Flip the kernel Flipping ensures the kernel detects edges in the correct direction. Without the flip, left and right would be reversed
    kh, kw = k.shape #2: Get kernel size and calculate padding needed.
    ph, pw = kh // 2, kw // 2
    if arr.ndim == 3: #ndim = number of dimensions grayscale ndim = 2, color ndim = 3
        padded = np.pad(arr, ((ph, ph), (pw, pw), (0, 0)), mode='constant')  #padding top bottom left right, (0,0) means don't pad channels 'constant' = zeros by default
        out = np.zeros_like(arr, dtype=np.float64)  #np.zeros_like() creates a new array filled with zeros
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for c in range(arr.shape[2]):
                    out[i, j, c] = np.sum(padded[i:i+kh, j:j+kw, c] * k_flipped)  #Get rows from i to i+kernel_height --Get columns from j to j+kernel_width --Get only channel c (red, green, or blue)
                    #The Problem: Out of Bounds! if no padding
    else:
        padded = np.pad(arr, ((ph, ph), (pw, pw)), mode='constant')
        out = np.zeros_like(arr, dtype=np.float64)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * k_flipped)
                #This line does the convolution for one pixel. It extracts a small square patch around pixel (i, j) from the padded image,
                # multiplies it with the kernel, sums all the results, and stores that sum as the new pixel value


    return out


if __name__ == '__main__':
    k = make_kernel(3, 1)   # todo: find better parameters
    
    # TODO: chose the image you prefer
    #im = np.array(Image.open('data/input1.jpg'))
    im = np.array(Image.open('data/input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))
    
    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
    blurred = slow_convolve(im, k)
    mask = im - blurred  #Subtracts the blurred image from the original This gives us the lost details (what was lost when blurring)
    sharpened = im + mask #Adds the mask (sharp details) back to the original image This enhances the edges and makes them more pronounced
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    Image.fromarray(sharpened).save('output.jpg') #convert numpy image back to img and save it
