import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('data/0_out.png',0)
print(np.shape(img))
import  os

path = '../../../datasets/Boston/'

print(os.listdir(path))


"""

kernelSize = 70
kernel = np.ones((kernelSize,kernelSize),np.uint8)
mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
print(mask)
plt.imsave('subplots.png',mask)


imgColor = cv2.cvtColor(cv2.imread('data/0.png'),cv2.COLOR_BGR2RGB)
plt.imsave('color.png',imgColor)

index = mask == 255

print(index)
print(type(mask))

sky = np.zeros(np.shape(imgColor),dtype=np.uint8)
sky[index,:] = imgColor[index,:]



histIm = imgColor[index,:]

plt.imsave('sky.png', sky)
print(np.shape(imgColor),type(imgColor), np.shape(mask), type(mask))
color = ('r','g','b')
print(type(sky),np.shape(sky))
histArray = []
for channel, col in enumerate(color):
    hist = cv2.calcHist([imgColor],[channel],mask,[255],[1,256])
    hist = np.array(hist,dtype=int).flatten()
    print(hist)
    histArray.append(hist)
    plt.plot(hist,color=col)
    #plt.xlim([1,256])

plt.savefig('hist')

print(np.shape(histArray))

"""