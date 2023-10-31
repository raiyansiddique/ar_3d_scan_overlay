import cv2
import numpy as np
def homography(img_src, pts_src):
    img_src = cv2.imread(img_src)
    im_dst = cv2.imread('square.jpg')
    pts_dst = np.array([[ 27,  27],
       [573,  27],
       [573, 573],
       [ 27, 573]])

    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(img_src, h, (im_dst.shape[1],im_dst.shape[0]))
 
    cv2.imshow("Warped Source Image", im_out)
 
    cv2.waitKey(0)
pts_src = np.array([[523,  217],
       [814,  217],
       [814, 382],
       [ 523, 382]])
homography("image1.jpg", pts_src)