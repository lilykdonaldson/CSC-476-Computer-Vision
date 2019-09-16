#Donaldson, Lily 4459300
#Homework 1
#Computer Vision 476

import numpy as np
from numpy import matlib
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import scipy


#1.	Basic Matrix/Vector Manipulation (20 points)
#a)	Define Matrix M and Vectors a, b, c in Python. You should use Numpy.
M = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9],[0, 2, 2]])

a = np.array([1, 1, 0])
b = np.array([-1, 2, 5])
c = np.array([0, 2, 3, 2])

#b)	Find the dot product of vectors a and b. 
	#Save this value to the variable aDotb
aDotb = np.dot(a, b)
#print(aDotb)
#dot product (a*b = [1])

#c)	Find the element-wise product of a and b
elementWise = np.multiply(a,b)
#print(elementWise)
#element-wise product = [-1  2  0]

#d) Find ((a^T)b)*(M*a)
moreMatrix = np.dot(a.T, b) * np.dot(M, a)
#print(moreMatrix)
# = [ 3  9 15  2]

#e)	Without using a loop, multiply each row and of M element-wise by a. 
#(Hint: the function repmat() may come in handy). 
#numpy.matlib.repmat(matrix, #repeaat1st, #repeat2nd)
repeatMe = np.matlib.repmat(a, 4, 1)
moreMoreMatrix = np.multiply(repeatMe, M)
#print(moreMoreMatrix)
# = [[1 2 0]
# [4 5 0]
# [7 8 0]
# [0 2 0]]

#f)	Without using a loop, sort all of the values of the new M from (e)
# in increasing order and plot them in your report. 
#ndarray.flatten(order='C')       A copy of the input array, flattened to one dimension.
repeatMe = moreMoreMatrix.flatten('C')
#print(repeatMe)
#[1 2 0 4 5 0 7 8 0 0 2 0]
repeatMe.sort()
#print(repeatMe)
#[0 0 0 0 0 1 2 2 4 5 7 8]
moreMoreMatrix = np.flipud(repeatMe)
#print(moreMoreMatrix)
#[8 7 5 4 2 2 1 0 0 0 0 0]


#---------------------------------------------------------------------
#2. Basic Image Manipulations (20 points)
#a)	Read in the images, image1.jpg and image2.jpg
dog = cv2.imread('image1.jpg',0)
cat = cv2.imread('image2.jpg',0)
#b)	Convert the images to double precision and rescale 
#them to stretch from minimum value 0 to maximum value 1.
startImage1 = np.float64(dog)
startImage2 = np.float64(cat)
zeroesD = np.zeros((256, 256))
myDog = cv2.normalize(startImage1, zeroesD, 0, 1, cv2.NORM_MINMAX)
zeroesD = np.zeros((256, 256))
myCat = cv2.normalize(startImage2, zeroesD, 0, 1, cv2.NORM_MINMAX)
cv2.imshow('Dog Double',myDog)
cv2.waitKey(0)
cv2.imshow('Cat Double',myCat)
cv2.waitKey(0)

#c)	Add the images together and re-normalize them to have minimum, 
#value 0 and maximum value 1. Display this image in your report. 
zeroesD = np.zeros((256, 256))
catDog = cv2.normalize((myDog+myCat), zeroesD, 0, 1, cv2.NORM_MINMAX)
cv2.imshow('Cat-Dog',catDog)
cv2.waitKey(0)

#d)	Create a new image such that the left half of the image is the left half 
    #of image1 and the right half of the image is the right half of image 2. 
#numpy.concatenate((a1, a2, ...), axis=0, out=None)
#0 through 256 tall, 0 to width/2 wide
cropDog = myDog[0:256, 0:256/2]
#0 through 256 tall, width/2 to end wide
cropCat = myCat[0:256, 256/2:]
#add 'em
splitCatDog = np.concatenate((cropDog, cropCat), axis=1)
cv2.imshow('Cat-Dog Split',splitCatDog)
cv2.waitKey(0)

#e)	Using a for loop, create a new image such that every odd numbered row is the 
#corresponding row from image1 and the every even row is the corresponding row from 
#image2 (Hint: Remember that indices start at 0 and not 1 in Python). 
#Display this image in your report.
#empty 256x256
fillMeUp = np.zeros((256, 256))
for i in range(256):
    if i%2 == 1:
    	#fill the odds
        fillMeUp[i] = myDog[i]
    else:
    	#fill the evens
        fillMeUp[i] = myCat[i]
cv2.imshow('Woahhhh Cat-Dog',fillMeUp)
cv2.waitKey(0)
#Wider cat test
# fillMeUp = np.zeros((256, 256))
# for i in range(256):
#     if i%5 == 0:
#         fillMeUp[i] = myDog[i]
#     else:
#         fillMeUp[i] = myCat[i]
# cv2.imshow('Woahhhh Cat-Dog',fillMeUp)
# cv2.waitKey(0)

#f)	Accomplish the same task as part e) without using a for-loop 
#(the functions reshape and repmat may be helpful here).
#empty 256x256
fillMeUpAgain = np.zeros((256, 256))
#get the odds
doggy = myDog[1::2]
#get the evens
catty = myCat[::2]
#fill the odds
fillMeUpAgain[1::2] = doggy
#fill the evens
fillMeUpAgain[::2] = catty
cv2.imshow('Woahhhh Cat-Dog Again',fillMeUpAgain)
cv2.waitKey(0)

#g)	Convert the result from part f) to a grayscale image. 
#Display the grayscale image with a title in your report.
#gray = cv2.cvtColor(fillMeUpAgain, cv2.COLOR_BGR2GRAY)
cv2.imshow("I'm already Gray",fillMeUpAgain)
cv2.waitKey(0)

#---------------------------------------------------------------------
import os
import skimage as io
from skimage import io
#3. Compute the average face (20pts)
#b)	Call numpy.zeros to create a 250 x 250 x 3 float64 tensor to hold the results
zeroes3 = np.zeros((250, 250, 3), dtype=np.float64)
#c)	Read each image with skimage.io.imread, convert to float and accumulate. 
ifw = os.listdir('ifw')
gerhards = []
gerhards = np.float64(np.array([np.array(io.imread('ifw/' + thisGerhard)) for thisGerhard in ifw]))
#print(gerhards)
#d)	Write the averaged result with skimage.io.imsave.  Post your resulted image in the report. 
averaged = np.array(np.mean(gerhards, axis=0), dtype=np.uint8)
io.imsave('mega_gerhard.jpg', averaged)
mega = cv2.imread('mega_gerhard.jpg',0)
cv2.imshow("Mixed Up Gerhards",mega)
cv2.waitKey(0)
