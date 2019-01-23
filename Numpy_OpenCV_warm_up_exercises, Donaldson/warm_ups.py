#CSC 476 Numpy OpenCV warm up exercises
#Lily Donaldson
#1/16/19

import numpy as np
import cv2
import matplotlib.pyplot as plt


#Numpy (7)
#1.	Create a 3x3 identity matrix
anIdentity = np.identity(3)
	#print(anIdentity)

#2.	Create a 3x3 array with random values 
aRandom = np.random.random((3,3))
	#print(aRandom)

#3.	Create a 10x10 array with random values and find the min and max values
aTenRandom = np.random.random((10,10))
aTenRandomMin = aTenRandom.min()
aTenRandomMax = aTenRandom.max()
	#print(aTenRandom)
	#print(aTenRandomMin,aTenRandomMax)

#4.	How to add a border (filled with 0s) around an existing array
newARandom = np.pad(aRandom,(1,1),'constant', constant_values=(0))
		#print(newARandom)

#5.	Create a random vector of size 40 and find the mean value 
findTheMean = np.random.random(40)
theMean = findTheMean.mean()
	#print(theMean)

#6.	Create a checkerboard 8x8 matrix using the tile function 
tiling = np.tile((['X','O'],['O','X']),(4,4))
	#print(tiling)

#7.	Create a vector of 100 uniform distributed values between 0 and 1.  
uniformDist = np.random.uniform(0,1,(100,100))
	#print(uniformDist)

#-----------------------------------------------------------------------------------------------------------------------

#Matplotlib (3)
#Using Numpy, Create a vector of 1000 random values drawn from a normal distribution with a mean of 0 and a standard deviation of 0.5
matplotlib = np.random.normal(0,0.5,1000)
	#print(matplotlib)

#Using matplotlib and and create such a plot:
plt.plot(matplotlib)
plt.show()

#-----------------------------------------------------------------------------------------------------------------------

#OpenCV (3) 
#1.	Save an image as .jpg in your folder and load it the image in grasyscale. Display the image and when the user click any key, the image disappear. 
goldenDuck = cv2.imread('duck.jpg')
noLongerGoldenDuck = cv2.cvtColor(goldenDuck, cv2.COLOR_BGR2GRAY)
cv2.imshow('I am no longer golden duck. Sad.', noLongerGoldenDuck)
cv2.waitKey(0)
cv2.destroyAllWindows()

#2.	Using cv2.imwrite() to save the image as .png
cv2.imwrite('grey_duck.png',noLongerGoldenDuck)

#3.	Find the brightest and darkest pixels value of the grayscale image. 
minValue, maxValue, useless1, useless2 = cv2.minMaxLoc(noLongerGoldenDuck)
	#print(maxValue,minValue)
