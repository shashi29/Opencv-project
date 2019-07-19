from panorama_saw import Panaroma
import imutils
import cv2
import sys
import glob

PATH = sys.argv[-1]

images = []
for image_name in sorted(glob.glob(PATH+'/*.jpg')):
    images.append(cv2.imread(image_name))

no_of_images = len(images)
print("Length of images :",no_of_images)


for i in range(no_of_images):
    images[i] = imutils.resize(images[i], width=400)

for i in range(no_of_images):
    images[i] = imutils.resize(images[i], height=400)


result = []
panaroma = Panaroma()

if no_of_images % 2 == 0:
    if no_of_images == 2:
        (result, matched_points) = panaroma.image_stitch([images[0], images[1]], match_status=True,combine=0)

    if no_of_images == 4:
        #no_of_images == 4:
        print("How it is working")
        (result_top, matched_points) = panaroma.image_stitch([images[0], images[2]], match_status=True,combine=0)
        (result_bottom, matched_points) = panaroma.image_stitch([images[1], images[3]], match_status=True,combine=0)
        (result,matched_points) = panaroma.image_stitch([result_top,result_bottom], match_status=True,combine=1)

    else :
        (result_top, matched_points) = panaroma.image_stitch([images[no_of_images-4], images[no_of_images-2]], match_status=True,combine=0)
        for i in range(no_of_images//4):
            try:
                (result_top, matched_points) = panaroma.image_stitch([images[no_of_images//2-2*(i+1)],result_top], match_status=True,combine=0)
            except:
                continue
        (result_bottom, matched_points) = panaroma.image_stitch([images[no_of_images-3], images[no_of_images-1]], match_status=True,combine=0)
        for i in range(no_of_images//4):
            try:
                (result_bottom, matched_points) = panaroma.image_stitch([images[(no_of_images//2-2*(i+1))+1],result_bottom], match_status=True,combine=0)
            except:
                continue
        (result,matched_points) = panaroma.image_stitch([result_top,result_bottom], match_status=True,combine=1)
        

if no_of_images > 2 and no_of_images % 2 != 0:
    (result, matched_points) = panaroma.image_stitch([images[no_of_images-2], images[no_of_images-1]], match_status=True,combine=0)
    for i in range(no_of_images - 2):
        (result, matched_points) = panaroma.image_stitch([images[no_of_images-i-3],result], match_status=True,combine=0)

   
#to show the got panaroma image and valid matched points
for i in range(no_of_images):
    cv2.imshow("Image {k}".format(k=i+1), images[i])

#cv2.imshow("Keypoint Matches", matched_points)
cv2.imshow("Panorama", result)

#to write the images
#cv2.imwrite("Matched_points.jpg",matched_points)
cv2.imwrite("Panorama_image.jpg",result)

cv2.waitKey(0)
cv2.destroyAllWindows()
