import cv2 as cv
import numpy as np
name = 'image002.png'
img = cv.imread(name, 0)
rows, cols = img.shape
crow, ccol = rows // 2 , cols // 2

size = min(rows, cols)
percentage = 1
size *= percentage
size = int(size)
if size % 2 == 0 :
    size -= 1

def HighPassGaussianFilter(sigma) :

    x = cv.getGaussianKernel(size, sigma)
    gaussian = x*x.T
    edgeval = gaussian[0][0]
    centerval = gaussian[size//2][size//2]
    gaussianrange =  centerval - edgeval
    gaussian = (gaussian - edgeval) / (gaussianrange)
    gaussian = 1 - gaussian
    return gaussian

img1 = img.copy()
img1 = img.astype('float64')
imgbefore = img
imgbefore = img.astype('float64')
sig = 40
gaussianfilter = HighPassGaussianFilter(sig)
# optimal value :
# image002:  1
# image006:  1.15
# cafe3:  4.5
# bookstore: 4.56

for i in range(1) :
    minval = np.min(img1[:,:])
    maxval = np.max(img1[:,:])
  
    img1[:,:] = 1 + (img1[:,:] - minval)/(maxval - minval) * 254
    imgbefore[:,:] = img1[:,:]

    img1[:,:] = np.log(img1[:,:])
    f = np.fft.fft2(img1[:,:])
    fshift = np.fft.fftshift(f)
    fshift[crow-size//2:crow+size//2+1,ccol-size//2:ccol+size//2+1] *= gaussianfilter
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    img1[:,:] = np.exp(img_back)
    
    minval = np.min(img1[:,:])
    maxval = np.max(img1[:,:])
    img1[:,:] = (img1[:,:] - minval)/(maxval - minval) * 255


img1 = img1.astype('uint8') 

equalized = cv.equalizeHist(img1)
# equalized = 16 * np.sqrt(img1)

imgbefore = imgbefore.astype('uint8')

cv.imwrite(name[:-4] + ' sigma=' + str(sig) + ' filtered.jpg',img1)
cv.imwrite(name[:-4] + ' sigma=' + str(sig) + ' filtered&equalized.jpg',equalized)

