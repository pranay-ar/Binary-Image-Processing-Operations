import numpy as np
import cv2 as cv2
#Reading the image
img = cv2.imread('regionfilling.png',0)
kernel=np.ones((3,3))
####Erosion function####
def erosion(image,kernel):
    image=image//255
    #Padding with zeros at the boundary
    o_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    image=np.zeros(image.shape)
    rows=len(o_image)
    cols=len(o_image[0])
    krows=len(kernel)
    kcols=len(kernel[0])
    #Convolving the kernel over the image
    for i in range(rows-2):
        for j in range(cols-2):
            counter=0
            for r in range(krows):
                for c in range(kcols):
                    if(o_image[i+r][j+c]==1):
                        counter=counter+1
            if(counter==krows*kcols):
                image[i][j]=1
    return image*255
#Dilation Function
def dilation(image,kernel):
    image=image//255
    #Padding with zeros at the boundary
    o_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    image=np.zeros(image.shape)
    rows=len(o_image)
    cols=len(o_image[0])
    krows=len(kernel)
    kcols=len(kernel[0])
    for i in range(rows-2):
        for j in range(cols-2):
            counter=0
            for r in range(krows):
                for c in range(kcols):
                    if(o_image[i+r][j+c]==1):
                        counter=counter+1
            if(counter>=1):
                image[i][j]=1
    return image*255

###closing function####
img_noise1=dilation(img,kernel)
img_noise1=erosion(img_noise1,kernel)

for _ in range(7):
    img=dilation(img,kernel)

cv2.imwrite('filling.jpg',img)

