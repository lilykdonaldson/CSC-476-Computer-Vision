import numpy as np
import scipy.signal as sig
from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from GeneratePyramid import *
plt.axis('off')
orange = misc.imread('apple.jpg',flatten=1)
apple = misc.imread('orange.jpg',flatten=1)
#reconstruct(img)
oG,oL = pyramids(orange)
aG,aL = pyramids(apple)

blendEmLap(orange,apple)

img1 = misc.imread('testplot1.png',flatten=1)
img2 = misc.imread('testplot2.png',flatten=1)
blended = img1+img2
fig, ax = plt.subplots()
ax.imshow(blended,cmap='gray')

plt.savefig('lapfruit.png')
plt.show()

blendEmGaus(orange,apple)

img1 = misc.imread('testplot1.png',flatten=1)
img2 = misc.imread('testplot2.png',flatten=1)
blended = img1+img2
fig, ax = plt.subplots()
ax.imshow(blended,cmap='gray')
plt.savefig('gausfruit.png')
plt.show()



#-----------------------------
kristof = misc.imread('kristof.jpg',flatten=1)
duck = misc.imread('duck.jpg',flatten=1)
oG,oL = pyramids(kristof)
aG,aL = pyramids(duck)

blendEmLap(kristof,duck)

img1 = misc.imread('testplot1.png',flatten=1)
img2 = misc.imread('testplot2.png',flatten=1)
blended = img1+img2
fig, ax = plt.subplots()
ax.imshow(blended,cmap='gray')
plt.show

blendEmGaus(kristof,duck)

img1 = misc.imread('testplot1.png',flatten=1)
img2 = misc.imread('testplot2.png',flatten=1)
blended = img1+img2
fig, ax = plt.subplots()
ax.imshow(blended,cmap='gray')
plt.savefig('duckkristof.png')
plt.show()

#-----------------------------
twin1 = misc.imread('twin1.jpg',flatten=1)
twin2 = misc.imread('twin2.jpg',flatten=1)
oG,oL = pyramids(twin1)
aG,aL = pyramids(twin2)

blendEmLap(twin1,twin2)

img1 = misc.imread('testplot1.png',flatten=1)
img2 = misc.imread('testplot2.png',flatten=1)
blendedSave = img1+img2
fig, ax = plt.subplots()
ax.imshow(blendedSave,cmap='gray')
plt.savefig('laptwins.png')
plt.show()

blendEmGaus(twin1,twin2)

img1 = misc.imread('testplot1.png',flatten=1)
img2 = misc.imread('testplot2.png',flatten=1)
blendedSave2 = img1+img2
fig, ax = plt.subplots()
ax.imshow(blendedSave2,cmap='gray')
plt.savefig('gaustwins.png')
plt.show()

finalBlend = blendedSave+blendedSave2
fig, ax = plt.subplots()
ax.imshow(finalBlend,cmap='gray')
plt.savefig('twinsboth.png')
plt.show()
#-----------------------------
sister = misc.imread('taylor.jpg',flatten=1)
brother = misc.imread('jack.jpg',flatten=1)
oG,oL = pyramids(sister)
aG,aL = pyramids(brother)

blendEmLap(sister,brother)
img1 = misc.imread('testplot1.png',flatten=1)
img2 = misc.imread('testplot2.png',flatten=1)
blendedSave3 = img1+img2

blendEmGaus(sister,brother)
img1 = misc.imread('testplot1.png',flatten=1)
img2 = misc.imread('testplot2.png',flatten=1)
blendedSave4 = img1+img2

finalish = finalBlend+blendedSave3+blendedSave4
fig, ax = plt.subplots()
ax.imshow(finalish,cmap='gray')
plt.savefig('siblingsall.png')
plt.show()


