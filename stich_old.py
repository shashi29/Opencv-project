from panorama import Panaroma
import imutils
import cv2
import sys
import glob

PATH = sys.argv[-1]

images = []
for image_name in sorted(glob.glob(PATH+'/*.jpg')):
    images.append(cv2.imread(image_name))

no_of_images = len(images)
#We need to modify the image resolution and keep our aspect ratio use the function imutils

for i in range(no_of_images):
    images[i] = imutils.resize(images[i], width=400)

for i in range(no_of_images):
    images[i] = imutils.resize(images[i], height=400)


panaroma = Panaroma()
if no_of_images==2:
    (result, matched_points) = panaroma.image_stitch([images[0], images[1]], match_status=True)
else:
    (result, matched_points) = panaroma.image_stitch([images[no_of_images-2], images[no_of_images-1]], match_status=True)
    for i in range(no_of_images - 2):
        (result, matched_points) = panaroma.image_stitch([images[no_of_images-i-3],result], match_status=True)

#to show the got panaroma image and valid matched points
for i in range(no_of_images):
    cv2.imshow("Image {k}".format(k=i+1), images[i])

cv2.imshow("Keypoint Matches", matched_points)
cv2.imshow("Panorama", result)

#to write the images
cv2.imwrite("Matched_points.jpg",matched_points)
cv2.imwrite("Panorama_image.jpg",result)

cv2.waitKey(0)
cv2.destroyAllWindows()
