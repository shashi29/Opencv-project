import numpy as np
import cv2
import imutils
import seaborn as sns

img = cv2.imread('3.jpg')
cv2.imshow('image',img)
#k = cv2.waitKey(0)
#if k == 27:         # wait for ESC key to exit
#    cv2.destroyAllWindows()
#elif k == ord('s'): # wait for 's' key to save and exit
#    cv2.imwrite('messigray.png',img)
#    cv2.destroyAllWindows()
    
#pre-processing for the image
icol = (0,114,23,255,255,66)    # Pipes

frameBGR = cv2.GaussianBlur(img, (7, 7), 0)
#frameBGR = cv2.medianBlur(img, 7)
#frameBGR = cv2.bilateralFilter(frameBGR, 15 ,75, 75)
#kernal = np.ones((15, 15), np.uint8)
#frameBGR = cv2.filter2D(frameBGR, -1, kernal)"""
	
# Show blurred image.
cv2.imshow('blurred', frameBGR)
	
# HSV (Hue, Saturation, Value).
# Convert the frame to HSV colour model.
hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)

# HSV values to define a colour range.
colorLow = np.array([0,114,23])
colorHigh = np.array([255,255,66])
mask = cv2.inRange(hsv, colorLow, colorHigh)
# Show the first mask
cv2.imshow('mask-plain', mask)
 
#kernel = np.ones((5,5),np.uint8)
#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal,iterations=1)
#mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal,iterations=1)
 
# Show morphological transformation mask
#cv2.imshow('mask', mask)

# Put mask over top of the original image.
result = cv2.bitwise_and(img, img, mask = mask)
cv2.imshow('result',result)

# find contours in the thresholded image
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("[INFO] {} unique contours found".format(len(cnts)))
 
# loop over the contours
k = 0
a = []

for (i, c) in enumerate(cnts):
	# draw the contour
	#print("The value for each c",c)
	((x, y), _) = cv2.minEnclosingCircle(c)
	a.append(cv2.contourArea(c))
	if(cv2.contourArea(c) > 500 and cv2.contourArea(c)<2500):
		cv2.putText(img, "#{}".format(k + 1), (int(x) - 10, int(y)),
    		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		k = k + 1
		print("The value for K",k)        
    	#print("Area of rectangle",cv2.contourArea(c))

		cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

print("Total number of pipes",k-1)
# show the output image
cv2.imshow("Image", img)
cv2.waitKey(0)

#Plot to see the average contours area distribution
a1 = sns.distplot(a,hist=True,kde=True,rug=True)
