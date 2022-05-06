from sys import platform
import cv2 as cv
from time import sleep

class CameraError(Exception): ...

OPTIMAL_TIME = 0.125 #The pause after the camera is turned on
OPTIMAL_COLOR = cv.COLOR_BGRA2GRAY if platform == "darwin" else cv.COLOR_BGR2GRAY #Default color depending on OS. BGRA in MacOS and BGR in Windows/Linux

cam = cv.VideoCapture(0) #turn on the camera
sleep(OPTIMAL_TIME)

prev = cv.cvtColor(cam.read()[1], OPTIMAL_COLOR) #quickly get one pic

while True:
    success, right = cam.read()

    if not success: raise CameraError("An error occured while reading from camera") #if couldn't get the image from camera successfully, throw an exception

    gray = cv.GaussianBlur(cv.cvtColor(right, OPTIMAL_COLOR), (31, 31), 0) #blur image to reduce noise and thus useless motion/contours 

    dif = cv.absdiff(gray, prev) #find the difference between the previous and this image, to detect motion

    thresh = cv.dilate(cv.threshold(dif, 30, 255, cv.THRESH_BINARY)[1], None, iterations=2) #convert it to binary image and dilute it

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #get all the contours
    if len(contours) == 0: continue #if no motion
    
    ctr = None
    max_ = 0
    #getting contour with the maximum area which in 99% cases is right
    for i in contours:
        area = cv.contourArea(i)
        if area >= max_:
            ctr = i
            max_ = area

    if max_ < 200: #check if it's actually right object to be detected
        prev = gray #set the comparison image for next frame this
        continue #continue to the next frame

    x,y,w,h = cv.boundingRect(ctr) #get x axis, y axis, width and height of the rectangle around maxed-contour
    cv.rectangle(right, (x, y), (x+w, y+h), (0, 255, 0), 2) #draw a rectangle of green color with 2 thickness

    cv.imshow("Motion Detector", right) #show the current frame

    prev = gray #again, set this as the image to be compared
    if cv.waitKey(1) & 0xFF == ord('q'): #exit the program if q is pressed
        break

cam.release() #turn camera off
cv.destroyAllWindows() #destroy any opened opencv window
