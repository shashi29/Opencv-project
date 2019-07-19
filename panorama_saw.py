import numpy as np
import imutils
import cv2


class Panaroma:

    def image_stitch(self, images, lowe_ratio=0.75, max_Threshold=4.0, match_status=False,combine=0):

        #detect the features and keypoints from SIFT
        (imageB, imageA) = images
        #imageB is 3rd and imageA is 4th 
        (KeypointsA, features_of_A) = self.Detect_Feature_And_KeyPoints(imageA)
        (KeypointsB, features_of_B) = self.Detect_Feature_And_KeyPoints(imageB)

        #got the valid matched points
        Values = self.matchKeypoints(KeypointsA, KeypointsB,features_of_A, features_of_B, lowe_ratio, max_Threshold)

        if Values is None:
            return None

        #to get perspective of image using computed homography
        (matches, Homography, status) = Values
        
        if combine == 0:
            result_image = self.getwarp_perspective(imageA,imageB,Homography)
            result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        #cv2.waitKey(0)
        if combine == 1:
            result_image = self.getwarp_perspective_vertical(imageA,imageB,Homography)
            result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
           
        cv2.imshow("How bad it combine", result_image)
        cv2.waitKey(0)
        # check to see if the keypoint matches should be visualized

        
        if match_status:
            vis = self.draw_Matches(imageA, imageB, KeypointsA, KeypointsB, matches,status)

            return (result_image, vis)
        
        return result_image

    def getwarp_perspective_vertical(self,imageA,imageB,Homography):
        val = imageA.shape[0] + imageB.shape[0]
        result_image = cv2.warpPerspective(imageA, Homography, (2*val , imageA.shape[1]))
        cv2.imshow("result image", result_image)
        cv2.imshow("image A", imageA)
        cv2.imshow("image B", imageB)
        print(result_image.shape,imageA.shape,imageB.shape)
        cv2.waitKey(0)
        return result_image

    def getwarp_perspective(self,imageA,imageB,Homography):
        val = imageA.shape[1] + imageB.shape[1]
        result_image = cv2.warpPerspective(imageA, Homography, (val , imageA.shape[0]))

        return result_image

    def Detect_Feature_And_KeyPoints(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image
        descriptors = cv2.xfeatures2d.SIFT_create()
        #descriptors = cv2.ORB_create()
        (Keypoints, features) = descriptors.detectAndCompute(image, None)

        Keypoints = np.float32([i.pt for i in Keypoints])
        return (Keypoints, features)

    def get_Allpossible_Match(self,featuresA,featuresB):

        # compute the all matches using euclidean distance and opencv provide
        #DescriptorMatcher_create() function for that
        match_instance = cv2.DescriptorMatcher_create("BruteForce")
        All_Matches = match_instance.knnMatch(featuresA, featuresB, 2)

        return All_Matches

    def All_validmatches(self,AllMatches,lowe_ratio):
        #to get all valid matches according to lowe concept..
        valid_matches = []

        for val in AllMatches:
            if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
                valid_matches.append((val[0].trainIdx, val[0].queryIdx))

        return valid_matches

    def Compute_Homography(self,pointsA,pointsB,max_Threshold):
        #to compute homography using points in both images

        (H, status) = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, max_Threshold)
        return (H,status)

    def matchKeypoints(self, KeypointsA, KeypointsB, featuresA, featuresB,lowe_ratio, max_Threshold):

        AllMatches = self.get_Allpossible_Match(featuresA,featuresB);
        valid_matches = self.All_validmatches(AllMatches,lowe_ratio)

        pointsA = []
        pointsB = []

        if len(valid_matches) > 4:
            # construct the two sets of points
            pointsA = np.float32([KeypointsA[i] for (_,i) in valid_matches])
            pointsB = np.float32([KeypointsB[i] for (i,_) in valid_matches])

            (Homograpgy, status) = self.Compute_Homography(pointsA, pointsB, max_Threshold)
            #print("Valid matches for A:",pointsA)
            #print("Valis matches for B:",pointsB)

            return (valid_matches, Homograpgy, status)

            
        else:
            return None

    def get_image_dimension(self,image):
        (h,w) = image.shape[:2]
        return (h,w)

    def get_points(self,imageA,imageB):

        (hA, wA) = self.get_image_dimension(imageA)
        (hB, wB) = self.get_image_dimension(imageB)
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        #vis[0:hA, 0:wA] = imageA
        #vis[0:hB, wA:] = imageB
        vis[0:hA, wB:] = imageA
        vis[0:hB, 0:wB] = imageB

        return vis


    def draw_Matches(self, imageA, imageB, KeypointsA, KeypointsB, matches, status):
        #(hA,wA) = self.get_image_dimension(imageA)
        (hB,wB) = self.get_image_dimension(imageB)
        (hA,wA) = self.get_image_dimension(imageA)
        vis = self.get_points(imageA,imageB)
        
        ptAList = []
        ptBList = []
        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(KeypointsA[queryIdx][0]) + wB, int(KeypointsA[queryIdx][1]))
                ptB = (int(KeypointsB[trainIdx][0]), int(KeypointsB[trainIdx][1]))

                #cv2.line(vis, ptB, ptA, (0, 255, 0), 1)
                #print("Matching point between images",ptA[0],ptB[0])
                ptAList.append(ptA[0])
                ptBList.append(ptB[0])
                #cv2.imshow("Keypoint Matches", vis)
                #cv2.waitKey(0)
        max_overlap = max(ptAList)
        min_overlap = min(ptBList)
        cv2.line(vis, (max_overlap,0), (max_overlap,hB), (0, 0, 255), 5)
        cv2.line(vis, (min_overlap,0), (min_overlap,hA), (0, 0, 255), 5)
        cv2.imshow("Keypoint Matches", vis)
        cv2.waitKey(0)
        cv2.imwrite(str(len(KeypointsA))+"_Panorama_image.jpg",vis)
        return vis
