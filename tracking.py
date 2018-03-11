import numpy as np
import cv2
cap = cv2.VideoCapture(1)
#img=cv2.imread("/home/nikmul19/Desktop/Samples/-1474508814843.png")

cv2.namedWindow("trackbar")
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50

threshold=0
def nothing(x):
    pass
cv2.createTrackbar('mine', 'trackbar', 0, 200, nothing)
cv2.createTrackbar('blurr','trackbar',1,200,nothing)
flag=False

while cap.isOpened():
    ret, frame = cap.read()

    frame=cv2.bilateralFilter(frame,5,50,100)
    frame=cv2.flip(frame,1)

    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),(frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow("frame",frame)
    
    if flag==True:
        mask=fgbg.apply(frame)
        cv2.imshow("mask",mask)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("res",res)

        img = res[0:int(cap_region_y_end * frame.shape[0]),           int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        tr=cv2.getTrackbarPos('mine','trackbar')
        br=cv2.getTrackbarPos('blurr','trackbar')
        if br%2==0:
            br+=1
        gray=cv2.GaussianBlur(gray,(br,br),0)
        ret,thresh=cv2.threshold(gray,tr,255,cv2.THRESH_BINARY)
#----------------------------------------find max area contour---------------------------------------------------------------
        x,contours,y=cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        maxArea=-1
        for i in range(len(contours)):
            area=cv2.contourArea(contours[i])
            if area>maxArea:
                maxArea=area
                ind=i
        print(ind)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, contours[ind], 0, (0, 255, 0), 2)
#---------------------------------------find convex hull of max area contour---------------------------------------------------
        res=contours[ind]
        hull=cv2.convexHull(res)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
        
        cv2.imshow('img', thresh)
        cv2.imshow('drawing', drawing)

        
    k=cv2.waitKey(100)
    if k ==ord('b'):
        fgbg = cv2.createBackgroundSubtractorMOG2(0,bgSubThreshold)    
        flag=True
        print("cap")
        
cap.release()

cv2.destroyAllWindows()
