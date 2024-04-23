import numpy as np
import cv2 as cv
import time

start = time.time()
img2 = cv.imread('template.png',cv.IMREAD_GRAYSCALE) # trainImage
img1 = cv.imread('scene.pgm',cv.IMREAD_GRAYSCALE)   # queryImage

scale_percent = 30
width = int(img2.shape[1] * scale_percent / 100)
height = int(img2.shape[0] * scale_percent / 100)
dim = (width, height)
img2 = cv.resize(img2, dim, interpolation = cv.INTER_AREA)

cap = cv.VideoCapture(1)


while(True):
    ret, frame = cap.read()
    img1 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img1_color = frame

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)


    #img1M=cv.drawKeypoints(img1,kp1,img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #img2M=cv.drawKeypoints(img2,kp2,img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    good_without_list = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
            good_without_list.append(m)

    if len(good)>50:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_without_list ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_without_list ]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img1_color = cv.polylines(img1_color,[np.int32(dst)],True,(0,0,255),3, cv.LINE_AA)
    # cv.drawMatchesKnn expects list of lists as matches.
    imgMatches = cv.drawMatchesKnn(img1_color,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    cv.imshow('frame', imgMatches)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

