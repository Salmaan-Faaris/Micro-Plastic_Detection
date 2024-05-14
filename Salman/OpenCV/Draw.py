import cv2 as cv
import numpy as np

blank = np.zeros((800,1000,3), dtype='uint8')
cv.imshow("Blank",blank)

#Painting an image
blank[200:300, 300:400] = 0,255,0  # Green color
blank[300:400, 400:500] = 255,0,0  # Blue color
cv.imshow("green",blank)

# Drawing a Rectangle
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=3)
cv.rectangle(blank, (blank.shape[1]//2, blank.shape[0]//2), (800,1000),(0,255,0), thickness=3)
cv.imshow("Rectangle", blank)


# Drawing a Circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=3)
cv.imshow("circle", blank)


# Drawing a Line
cv.line(blank, (0,0),(blank.shape[1]//2, blank.shape[0]//2), (255,255,255), thickness=3)
cv.imshow("Line", blank)


# Adding Text to the Image
cv.putText(blank, "Hi Iam Salman",(0,400), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=5)
cv.imshow("Text", blank)
cv.waitKey(0)