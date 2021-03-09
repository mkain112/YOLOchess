import cv2
import numpy as np
path = '/home/mitch/Desktop/Screenshot from 2020-06-15 20-59-01.png'


img = cv2.imread(path)
cv2.imshow('original', img)
cv2.waitKey(0)
bl=[93,858]
br=[770,840]
tr=[728,6]
tl=[97,27]
width,height= 500,500
pts1= np.float32([tl,tr,bl,br])
pts2= np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
img2=cv2.warpPerspective(img,matrix,(width,height))
cv2.imshow('result', img2)

cv2.imwrite('RESULT.jpg',img2)
cv2.waitKey(0)