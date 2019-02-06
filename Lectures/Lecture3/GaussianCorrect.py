import numpy as np
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
def gaussian_kernel_1d(size):
	size = int(size)
	x = np.mgrid[-size:size+1]
	g = np.exp(-(x**2/float(size)))
	return g
 
# Make the Gaussian by calling the function
size = 3
gaussian_kernel_array = gaussian_kernel_1d(size)
g_2 = np.outer(np.transpose(gaussian_kernel_array),gaussian_kernel_array)
size = 1
gaussian_kernel_array = gaussian_kernel(size)
g_3 = np.outer(np.transpose(gaussian_kernel_array),gaussian_kernel_array)

plt.imshow(g_2, cmap=plt.get_cmap('jet'), interpolation='nearest')
plt.colorbar()
plt.show()
plt.imshow(g_3, cmap=plt.get_cmap('jet'), interpolation='nearest')
plt.colorbar()
plt.show()