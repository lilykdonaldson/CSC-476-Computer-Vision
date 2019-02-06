import numpy as np
from scipy import signal

# 1D conv
x = np.array([0,1,2,3,4])
w = np.array([1,-1,2])
print(x)
res = np.convolve(x,w, 'valid')
print(res)
res = np.convolve(x,w, 'full')
print(res)
res = np.convolve(x,w, 'same')
#print(res)

# # 2D convolution
# # 2D conv needs scipy
x = np.array([[1, 2,3], [4,5,6],[7,8,9]])
print(x.shape)
w = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
print(w.shape)

f = signal.convolve2d(x,w,'same')
print(f)

# # 2D convolution with sharpening

# x = np.array([[0, 0, 0, 0, 0], [0, 50, 50, 50, 0],[0, 50, 50, 50, 0],[0, 50, 50, 50, 0], [0, 0, 0, 0, 0]])
# print(x.shape)
# w1 = np.array([[0,0,0],[0,2,0], [0,0,0]])
# w0 = np.array([[1.0/9,1.0/9,1.0/9],[1.0/9,1.0/9,1.0/9],[1.0/9,1.0/9,1.0/9]])
# w = w1-w0
# print(w.shape)

# f1 = signal.convolve2d(x,w,'valid')
# print(f1)
# print(f1.shape)

#perform unsharp mask by using only ONCE the convolution commad

