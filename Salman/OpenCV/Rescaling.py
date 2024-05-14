# Rescaling an image
import cv2 as cv
img = cv.imread('Frozen Rose.jpg')
cv.imshow("Frozen rose", img)

def rescaleImg(img,scale=0.50):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(img,dimensions, interpolation = cv.INTER_AREA)

resized_image = rescaleImg(img)
cv.imshow("Resized Image",resized_image)

cv.waitKey(0)


# Rescaling a video
capture = cv.VideoCapture('ride.mp4')
    
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)    

while True:
    isTrue, frame = capture.read()
    cv.imshow("Original Video",frame)
    resized_frame = rescaleFrame(frame)
    cv.imshow("Resized Frame",resized_frame)
    if cv.waitKey(5) & 0xFF == ord('d'):
        break
    
capture.release
cv.destroyAllWindows()
