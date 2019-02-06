import numpy as np
from math import pi, sqrt, exp
import matplotlib.pyplot as plt

def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y)))
    return g / g.sum()

def gauss1(n=11,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

# Make the Gaussian by call ing the function
size = 5
gaussian_kernel_array = gaussian_kernel(size)
gaussian_kernel_array2 = gauss1()
res = np.outer(np.transpose(gaussian_kernel_array2),gaussian_kernel_array2)
#print(res);
#print(gaussian_kernel_array);
plt.imshow(gaussian_kernel_array, cmap=plt.get_cmap('jet'), interpolation='nearest')
plt.colorbar()
plt.show()
plt.imshow(res, cmap=plt.get_cmap('jet'), interpolation='nearest')
plt.colorbar()
plt.show()


