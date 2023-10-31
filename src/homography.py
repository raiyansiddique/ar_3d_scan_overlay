import cv2
import numpy as np

def homography(img_src, pts_src):
    img_src = cv2.imread(img_src)
    img_dst = np.zeros((8, 8), dtype=np.uint8)
    img_dst = cv2.merge((img_src, img_src, img_src))

    # im_dst = cv2.imread('square.jpg')
    # if im_dst is None:
    #     print("Error: Could not read the destination image.")
    #     return

    pts_dst = np.array([[ 0,  0],
                        [7,  0],
                        [7, 7],
                        [ 0, 7]])

    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(img_src, h, (img_dst.shape[1],img_dst.shape[0]))

    cv2.imshow("Warped Source Image", im_out)
    cv2.waitKey(0)

# Example usage
pts_src = np.array([[ 94,  97],
       [823,  97],
       [823, 826],
       [ 94, 826]])

img_src = "tag_36h11.png"



homography(img_src, pts_src)
