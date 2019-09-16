#Lily Donaldson 4459300
#HW2
#Computer Vision 476

#imports
import math
import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage
from scipy.ndimage import filters


#1.1-1.3
hat = cv2.imread('images/lowcontrast.jpg',0)
cat = cv2.imread('images/cheetah.png',0)

blurry_woman = ndimage.gaussian_filter(hat, 7)
blurry_cheetah = ndimage.gaussian_filter(cat, 7)

cv2.imshow("Blurred woman",blurry_woman)
cv2.waitKey(0)
cv2.imshow("Blurred cat",blurry_cheetah)
cv2.waitKey(0)

#1.4
dftHat = np.fft.fft2(hat)
dftHatShifted = np.fft.fftshift(dftHat)
#mag spectrum
mag_hat = np.log(np.abs(dftHatShifted))

dftCat = np.fft.fft2(cat)
dftCatShifted = np.fft.fftshift(dftCat)
#mag spectrum
mag_cat = np.log(np.abs(dftCatShifted))


plt.figure(1, figsize=(15, 5))

plt.subplot(1, 4, 2)
plt.title('MAG CAT', fontsize=10)
plt.imshow(mag_cat, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('MAG HAT', fontsize=10)
plt.imshow(mag_hat, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('HAT', fontsize=10)
plt.imshow(hat, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 1)
plt.title('CAT', fontsize=10)
plt.imshow(cat, cmap=plt.cm.gray)
plt.axis('off')

plt.show()







#-------------------------------------------------------

#2
#np.histogram(array_like,num_of_equal_width_bins)
#Compute the histogram of a set of data.
#np.arrange(start,stop,set)
#evenly spaced values
frequencies, bins = np.histogram(hat, bins=np.arange(-0.5, 255.1, 0.5))

intensities = bins[1:bins.size]
# cumulative distribution function (image's accumulated normalized histogram)
cdf = np.cumsum(frequencies)
# compute compensation transfer function (normalized CDF)
intensityMap = cdf/np.float32(cdf[-1]) * 255
#get target; 5%
target = cdf[-1] * 0.05
#ready
black = 0
i = 0
white = 0
j = frequencies.size - 1

#push full black
while (black < target):
    black += frequencies[i]
    intensityMap[i] = 0
    i += 1
#push full white
while (white < target):
    white += frequencies[j]
    intensityMap[j] = 255
    j -= 1

#finish
#get non pure black and pure white pixels
leftover = hat.size - (black + white)
tolerance = leftover*0.05
x = 0
while(x < tolerance):
    x = 0
    leftover = np.sum(frequencies[i:(j+1)])
    for k in range(i, j+1):
        ceiling = leftover/(j+1-i)
        if frequencies[k] > ceiling:
            x += frequencies[k] - ceiling
            frequencies[k] = ceiling

intensityMap[i:(j+1)] = np.cumsum(frequencies[i:(j+1)]).astype(float)/leftover*255.0 + 255*(float(black)/hat.size)
new_hat = np.interp(hat, intensities, intensityMap)

dftHist = np.fft.fft2(new_hat)
dftHistShift = np.fft.fftshift(dftHist)
#mag spectrum
mag_hist = np.log(np.abs(dftHistShift))

plt.figure(1, figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title('Old Image', fontsize=10)
plt.imshow(hat, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Old Mag', fontsize=10)
plt.imshow(mag_hat, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('New Image', fontsize=10)
plt.imshow(new_hat, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Equalized', fontsize=10)
plt.imshow(mag_hist, cmap=plt.cm.gray)
plt.axis('off')

plt.show()

frequencies, bins = np.histogram(hat, bins=np.arange(-0.5, 255.1, 0.5))

intensities = bins[1:bins.size]
# cumulative distribution function (image's accumulated normalized histogram)
cdf = np.cumsum(frequencies)
# compute compensation transfer function (normalized CDF)
intensityMap = cdf/np.float32(cdf[-1]) * 255
#get target; 5%
target = cdf[-1] * 0.4
#ready
black = 0
i = 0
white = 0
j = frequencies.size - 1

#push full black
while (black < target):
    black += frequencies[i]
    intensityMap[i] = 0
    i += 1
#push full white
while (white < target):
    white += frequencies[j]
    intensityMap[j] = 255
    j -= 1

#finish
#get non pure black and pure white pixels
leftover = hat.size - (black + white)
tolerance = leftover*0.4
x = 0
while(x < tolerance):
    x = 0
    leftover = np.sum(frequencies[i:(j+1)])
    for k in range(i, j+1):
        ceiling = leftover/(j+1-i)
        if frequencies[k] > ceiling:
            x += frequencies[k] - ceiling
            frequencies[k] = ceiling

intensityMap[i:(j+1)] = np.cumsum(frequencies[i:(j+1)]).astype(float)/leftover*255.0 + 255*(float(black)/hat.size)
new_hat = np.interp(hat, intensities, intensityMap)

dftHist = np.fft.fft2(new_hat)
dftHistShift = np.fft.fftshift(dftHist)
#mag spectrum
mag_hist_high = np.log(np.abs(dftHistShift))
scipy.misc.imsave('images/high_contrast_hat_woman.jpg', new_hat)

plt.figure(1, figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title('Old Image', fontsize=10)
plt.imshow(hat, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Old Mag', fontsize=10)
plt.imshow(mag_hat, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Newer Image', fontsize=10)
plt.imshow(new_hat, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('High Contrast Equalized', fontsize=10)
plt.imshow(mag_hist_high, cmap=plt.cm.gray)
plt.axis('off')

plt.show()






#-------------------------------------------------------
#3
#set gaussian, box, and sobel
holdArray= np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
gaussian = 1.0/256*holdArray
gaussianX = np.array([1,4,6,4,1])
holdArray=np.array([1,4,6,4,1])
gaussianY = 1.0/256*holdArray
box = np.ones((5,5), dtype=np.float32)/25
boxX = np.array([1,1,1,1,1])
boxY = np.array([1,1,1,1,1])/25.0
sobel = np.array([[1,2,0,-2,-1],[4,8,0,-8,-4],[6,12,0,-12,-6],[4,8,0,-8,-4],[1,2,0,-2,-1]])
sobelX = np.array([1,2,0,-2,-1])
sobelY = np.array([1,4,6,4,1])

#convolve with gaussian,box, and sobel
CATgaussian = filters.convolve(cat, gaussian)
CATgaussianX = filters.convolve1d(cat, gaussianX, axis=1)
CATgaussianY = filters.convolve1d(cat, gaussianY, axis=0)
CATbox = filters.convolve(cat, box) 
CATboxX = filters.convolve1d(cat, boxX, axis=1)
CATboxY = filters.convolve1d(cat, boxY, axis=0)
CATsobel = filters.convolve(cat, sobel)
CATsobelX = filters.convolve1d(cat, sobelX, axis=1)
CATsobelY = filters.convolve1d(cat, sobelY, axis=0)


#Display
plt.figure(1, figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title('Original Cat', fontsize=10)
plt.imshow(cat, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Cat, Box Filter', fontsize=10)
plt.imshow(CATbox, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Cat, gaussian filter', fontsize=10)
plt.imshow(CATgaussian, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Cat, Sobel Filter', fontsize=10)
plt.imshow(CATsobel, cmap=plt.cm.gray)
plt.axis('off')

plt.show()




#-------------------------------------------------------
#4
#Original image
mag_hist_high = np.float64(misc.imread('images/high_contrast_hat_woman.jpg', flatten=1, mode='F'))
#scipy horizontal sobel filter (0)
horizontal = ndimage.sobel(mag_hist_high, 0)
#scipy vertical sobel filter (1)
vertical = ndimage.sobel(mag_hist_high, 1)
#returns hypotenuse with a and b scalars (combines all edges)
myEdges = np.hypot(horizontal, vertical)

#Display
plt.figure(1, figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.title('Original', fontsize=10)
plt.imshow(mag_hist_high, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 4, 2)
plt.title('horizontal edges', fontsize=10)
plt.imshow(horizontal, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 4, 3)
plt.title('vertical edges', fontsize=10)
plt.imshow(vertical, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 4, 4)
plt.title('ALL THE EDGES', fontsize=10)
plt.imshow(myEdges, cmap=plt.cm.gray)
plt.axis('off')
plt.show()



