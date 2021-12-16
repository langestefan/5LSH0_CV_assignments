import numpy as np
import cv2 as cv


def brute_force_featurematch(des1, des2):
    # brute_force = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    brute_force = cv.BFMatcher()
    no_of_matches = brute_force.match(np.float32(des1), np.float32(des2))

    no_of_matches = sorted(no_of_matches, key=lambda x: x.distance)
    return no_of_matches


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # cv2.xfeatures2d.SIFT_create(5)

    # import the images and greyscale them
    img = [0, 1, 2]
    img_gr = [0, 1, 2]
    img_kp = [0, 1, 2]
    keypoints = [0, 1, 2]
    descrip = [0, 1, 2]

    img[0] = cv.imread('all_souls_000002.jpg')
    img[1] = cv.imread('all_souls_000006.jpg')
    img[2] = cv.imread('all_souls_000013.jpg')

    # create SIFT features
    sift = cv.SIFT_create()

    for idx in range(3):
        img_gr[idx] = cv.cvtColor(img[idx], cv.COLOR_BGR2GRAY)
        keypoints[idx], descrip[idx] = sift.detectAndCompute(img_gr[idx], None)
        img_kp[idx] = cv.drawKeypoints(img_gr[idx], keypoints[idx], img[idx],
                                       flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # output image
    cv.imwrite('img_1_kp.jpg', img_kp[0])
    cv.imwrite('img_2_kp.jpg', img_kp[1])
    cv.imwrite('img_3_kp.jpg', img_kp[2])

    # brute force match 2 of the images
    no_matches = brute_force_featurematch(descrip[0], descrip[2])

    print(len(no_matches))
