import cv2
import numpy as np
img = cv2.imread("cat.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("image",img)

cv2.imshow("gray_image",gray)

#scaling
r,c=img.shape[:2]
scale_img=cv2.resize(img,(2*r,2*c),interpolation=cv2.INTER_CUBIC)
#cv2.imshow("RESIZE",scale_img)

#cropping
img_crop=img[100:3200,150:350]
cv2.imshow("crop",img_crop)
#%%
#translation
#M=np.float32([[1,0,100],[0,1,100]])
M=cv2.getRotationMatrix2D((c/2,r/2),90,1)
#tra_img=cv2.warpAffine(img,M,(c,r))
rot_img=cv2.warpAffine(img,M,(c,r))
#cv2.imshow("translation",tra_img)
cv2.imshow("rotation",rot_img)
#%%
#Thresholding
th_img=cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
cv2.imshow("THRESHOLD BINARY",th_img[1])
#%%
#Filters

#gaus_blur_img=cv2.GaussianBlur(img,(5,5),0)
median_blur_img=cv2.medianBlur(img,5)
#cv2.imshow("GaussianBlur",gaus_blur_img)
cv2.imshow("MedianBlur",median_blur_img)
#%%
#Morphological
ker=np.ones((5,5),np.uint8)
ero_img=cv2.dilate(th_img,ker,iterations=1)
cv2.imshow("Erosion",ero_img)
#%%
#edge detection
x_edge=cv2.Sobel(gray,-1,1,0,ksize=5)
y_edge=cv2.Sobel(gray,-1,0,1,ksize=5)
canny_edge=cv2.Canny(gray,100,200,3)
cv2.imshow("xedges",x_edge)
cv2.imshow("yedges",y_edge)
cv2.imshow("canny",canny_edge)
