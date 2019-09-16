#lecture 3
import numpy as np
import cv2
import matplotlib.pyplot as plt
#1 -2
#3 2
#*
#1 2 4
#1 3 1
#exp (1x1)+(-2x1) = -1
#-1 -4 2
# 5 12 14
a = np.array([[1,-2],
			  [3,2]])
b = np.array([[1,2,4],
			  [1,3,1]])
print(np.matmul(a,b))

"""
s1 3[1,1,1]
s2 4[0,1,1]
s3 5[0,0,1]

3,3,3 + 0,4,4 + 0,0,5
[3,7,12]

"""