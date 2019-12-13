import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import urllib.request
from pdf2image import convert_from_path, convert_from_bytes
from fpdf import FPDF
import GetIMUData as imu
import img2pdf

# Input PDF file name
input_pdf = 'test.pdf'

# Setting up variables to traverse slides
slide_number = 1
next_slide = False
previous_slide = False

refresh = False
help_mode = False

prev_output = []
nt_counter = 0

# font for displaying text (below)
font = cv2.FONT_HERSHEY_SIMPLEX

# Variables to add Aruco markers to file
# Aruco markers need to be offset as we need the full black square, 
# if right next to the edge, the bounding black box is not detected
# This is an artefact of the Aruco Code detetction library
x_offset = 10 
y_offset = 10
# Have to scale the aruco markers image down as they are too large
scale_percent = 50 # percent of original size    

aruco_25 = cv2.imread('markers/6x6_1000-25.jpg')
aruco_50 = cv2.imread('markers/6x6_1000-50.jpg')
aruco_75 = cv2.imread('markers/6x6_1000-75.jpg')
aruco_100 = cv2.imread('markers/6x6_1000-100.jpg')

# read the pdf file and make an image of each file
pdf = FPDF()
images = convert_from_bytes(open(input_pdf, 'rb').read())

print("Annotating Slides with Aruco Markers")
for page in images:
    page.save('slides/'+str(slide_number)+'_out.jpg', 'JPEG')
    slide = cv2.imread('slides/'+str(slide_number)+'_out.jpg')

    # inserting Aruco marker with ID 25 in TOP LEFT
    width = int(aruco_25.shape[1] * scale_percent / 100)
    height = int(aruco_25.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_aruco_25 = cv2.resize(aruco_25, dim, interpolation = cv2.INTER_AREA)
    slide[y_offset:y_offset+resized_aruco_25.shape[0], x_offset:x_offset+resized_aruco_25.shape[1]] = resized_aruco_25

    # inserting Aruco marker with ID 50 in TOP RIGHT
    width = int(aruco_50.shape[1] * scale_percent / 100)
    height = int(aruco_50.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_aruco_50 = cv2.resize(aruco_50, dim, interpolation = cv2.INTER_AREA)
    slide[y_offset:y_offset+resized_aruco_50.shape[0], slide.shape[1] - x_offset-resized_aruco_50.shape[1]:slide.shape[1]-x_offset ] = resized_aruco_50

    # inserting Aruco marker with ID 100 in BOTTOM LEFT
    width = int(aruco_100.shape[1] * scale_percent / 100)
    height = int(aruco_100.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_aruco_100 = cv2.resize(aruco_100, dim, interpolation = cv2.INTER_AREA)
    slide[slide.shape[0]-y_offset-resized_aruco_100.shape[0]: slide.shape[0]-y_offset, slide.shape[1] - x_offset-resized_aruco_100.shape[1]:slide.shape[1]-x_offset ] = resized_aruco_100

    # inserting Aruco marker with ID 75 in BOTTOM RIGHT
    width = int(aruco_75.shape[1] * scale_percent / 100)
    height = int(aruco_75.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_aruco_75 = cv2.resize(aruco_75, dim, interpolation = cv2.INTER_AREA)
    slide[slide.shape[0]-y_offset-resized_aruco_75.shape[0]: slide.shape[0]-y_offset, x_offset:x_offset+resized_aruco_75.shape[1]] = resized_aruco_75
    
    # Save all slides as individual images
    cv2.imwrite('slides/'+str(slide_number)+'_out.jpg', slide)
    slide_number = slide_number + 1
print("Annotation complete")

# Load all images in alphabetical order
slides = glob.glob('slides/*.jpg')
slides.sort()

# Set variables to help navigation of slides
slide_number = 0
max_slide_number = len(slides)

# Setting variables to capture coordinates of Aruco markers
# These will be set based on the camera feed
# TL: Top Left
# TR: Top Right
# BL: Bottom Left
# BR: Bottom Right
# Format: [X_coordinate, Y_coordinate, flag ]
# flag: 0 - X,Y coordinate are not usable and need to be set 
# flag: 1 - X,Y coordinate can be used
TL = [-1, -1, 0]    
TR = [-1, -1, 0]
BL = [-1, -1, 0]
BR = [-1, -1, 0]

# GT - Ground Truth
# These are set from the image of the slide
GT_TL = [-1, -1]
GT_TR = [-1, -1]
GT_BL = [-1, -1]
GT_BR = [-1, -1]

# IMU initialization code
imu.start_imu()

# for bright projector screens we want less exposure
print("Starting Stream")
urllib.request.urlopen('http://192.168.8.101/control?var=aec&val=0')
urllib.request.urlopen('http://192.168.8.101/control?var=aec_value&val=64') # tweak value for image exposure (higger -> more)
urllib.request.urlopen('http://192.168.8.101/control?var=framesize&val=7')    # tweak value for image size (higher -> bigger)
stream = urllib.request.urlopen('http://192.168.8.101:81/stream.mjpg')
print("Stream established")
bytes = bytes()
first_time = True

# ---------------------- CALIBRATION ---------------------------
# ------------------------DO NOT EDIT --------------------------
# termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# checkerboard of size (7 x 6) is used
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# iterating through all calibration images
# in the folder
images = glob.glob('calib_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find the chess board (calibration pattern) corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # if calibration pattern is found, add object points,
    # image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # Refine the corners of the detected corners
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# ------------------ ARUCO TRACKER ---------------------------
while (True):
    bytes += stream.read(1024)
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')

    # set on key press of 'd'
    if(next_slide is True):
        slide_number = slide_number + 1
        #  loop around to first slide if on last slide
        if (slide_number == max_slide_number):
            slide_number = 0
        next_slide = False
    if(previous_slide is True):
        slide_number = slide_number - 1
        #  loop around to last slide if on first slide
        if( slide_number == -1):
            slide_number = max_slide_number - 1
        previous_slide = False

    # set the current slide 
    present_slide=slides[slide_number]

    if a != -1 and b != -1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        if(jpg):
            # convert input of byte stream to format usable by opencv
            frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            open_cv_image = np.array(frame)
            try:
                frame = frame[:,:,::-1].copy()
            except:
                print('Got a bad frame')
                continue
        else:
            print('Bad frame')
            continue

        # if being run for the first time, get ground truth from the image of the slide
        if(first_time is  True):
            img = cv2.imread(present_slide)
            frame = cv2.cvtColor(img,0)

        height =  frame.shape[0]
        width = frame.shape[1]

        # cnetre point of aruco marker
        CP_X = 0 
        CP_Y = 0

        # operations on the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # set dictionary size depending on the aruco marker selected
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

        # detector parameters can be set here (List of detection parameters[3])
        parameters = aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 10
        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Aruco detection code gives the corners for the aruco markers, need to calculate the centre from this
        # This array has the corners for all the aruco id's detected in the frame
        if(corners):
            for ele in range(0, len(corners)):
                C1_X = (corners[ele][0][0][0])
                C1_Y = (corners[ele][0][0][1])
                C2_X = (corners[ele][0][1][0])
                C2_Y = (corners[ele][0][1][1])
                C3_X = (corners[ele][0][2][0])
                C3_Y = (corners[ele][0][2][1])
                C4_X = (corners[ele][0][3][0])
                C4_Y = (corners[ele][0][3][1])
                CP_X = ( C1_X + C3_X ) / (2) 
                CP_Y = ( C1_Y + C3_Y ) / (2)

                # For the first time, we have the static image from the slide and our ground truth
                # every other time we have the camera feed
                if(first_time is False):
                    if(ids[ele] == 25 and TL[2] == 0):
                        TL[0] = CP_X
                        TL[1] = CP_Y    
                        TL[2] = 1	# set the result as usable

                    if(ids[ele] == 50 and TR[2] == 0):
                        TR[0] = CP_X
                        TR[1] = CP_Y
                        TR[2] = 1

                    if(ids[ele] == 100 and BR[2] == 0):
                        BR[0] = CP_X
                        BR[1] = CP_Y
                        BR[2] = 1

                    if(ids[ele] == 75 and BL[2] == 0):
                        BL[0] = CP_X
                        BL[1] = CP_Y
                        BL[2] = 1
                else:
                    if(ids[ele] == 25):
                        GT_TL[0] = CP_X
                        GT_TL[1] = CP_Y

                    if(ids[ele] == 50):
                        GT_TR[0] = CP_X
                        GT_TR[1] = CP_Y

                    if(ids[ele] == 75):
                        GT_BL[0] = CP_X
                        GT_BL[1] = CP_Y

                    if(ids[ele] == 100):
                        GT_BR[0] = CP_X
                        GT_BR[1] = CP_Y
                    
                    GT_height =  frame.shape[0]
                    GT_width = frame.shape[1]

        if first_time is True:
            img_slide_fresh = cv2.imread(present_slide)
            first_time = False
            continue            

		# display actuvation of helper mode        
        img_slide = cv2.imread(present_slide)    
        if(help_mode == True):
            # H in Green 
            cv2.putText(img_slide, "H", (400,20), font, 1, (0,255,0),2,cv2.LINE_AA)
        else:
            # H in Red
            cv2.putText(img_slide, "H", (400,20), font, 1, (0,0,255),2,cv2.LINE_AA)

        if(refresh == True):
            img_slide = img_slide_fresh
            cv2.imwrite(present_slide, img_slide)
            refresh = False

        # When all four corner points have been detected, compute the transform matrix
        if( BR[2] == 1 and BL[2] == 1 and TR[2] == 1 and TL[2] == 1):


            pts_src = np.array([ [    TL[0],    TL[1] ], [    TR[0],    TR[1] ], [    BR[0],    BR[1] ], [    BL[0],    BL[1] ]    ])
            pts_dst = np.array([ [ GT_TL[0], GT_TL[1] ], [ GT_TR[0], GT_TR[1] ], [ GT_BR[0], GT_BR[1] ], [ GT_BL[0], GT_BL[1] ]    ])
            # compute the homographic matrix, h
            h, status =  cv2.findHomography(pts_src, pts_dst)
            im_dst = cv2.warpPerspective(frame, h, (GT_width, GT_height))

            window = cv2.namedWindow("HOMOGRPAHY", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("HOMOGRPAHY", 1800,1000)
            cv2.imshow("HOMOGRPAHY",im_dst)

            centre_point =  np.array([[[width/2, height/2]]], dtype=np.float32)
            output = cv2.perspectiveTransform(centre_point, h)
        
            if len(prev_output) == 0:
                prev_output.append(output)

            data = imu.get_imu_data()
            action = imu.get_action(imu.featurize_input(data))

            if(action != 3):
                print(action)
            # Circle
            if(action == 0):
                cv2.circle(img_slide,(output[0][0][0], output[0][0][1]),50,(255,0,0),5)
                cv2.imwrite(present_slide, img_slide)
            # Rectangle
            if(action == 1):
                cv2.rectangle(img_slide,(int(prev_output[0][0][0]-(len(data)*10)),int(prev_output[0][0][1]-(len(data)*5))), (int(prev_output[0][0][0]+25),int(prev_output[0][0][1]+25)),(0,255,0),3)
                cv2.imwrite(present_slide, img_slide)
            # Line
            if(action == 2):
                if (help_mode):
                    cv2.line(img_slide,(int(output[0][0][0]),int(output[0][0][1])), (int(prev_output[0][0][0]),int(output[0][0][1])), (0,0,255), 5)
                else:
                    cv2.line(img_slide,(int(prev_output[0][0][0]),int(prev_output[0][0][1])), (int(output[0][0][0]),int(output[0][0][1])), (0,0,255), 5)
                cv2.imwrite(present_slide, img_slide)
            
            window = cv2.namedWindow("TURTLE", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("TURTLE", 1800,1000)
            cv2.circle(img_slide,(int(output[0][0][0]), int(output[0][0][1])) ,10,(0,0,255),5)
            cv2.line(img_slide,(int(prev_output[0][0][0]),int(prev_output[0][0][1])), (int(output[0][0][0]),int(output[0][0][1])), (0,0,255), 5)
            cv2.putText(img_slide, "Tracking", (200,20), font, 1, (0,255,0),2,cv2.LINE_AA)
            cv2.imshow("TURTLE",img_slide)
            prev_output = output

            TL[2] = 0
            TR[2] = 0
            BR[2] = 0
            BL[2] = 0
        else:
        	# If all four corners are not being tracked, display a message on the screen saying the same
            if(nt_counter > 60):
                window = cv2.namedWindow("TURTLE", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("TURTLE", 1800,1000)
                cv2.putText(img_slide, "Not tracking", (200,20), font, 1, (0,0,255),2,cv2.LINE_AA)
                cv2.imshow("TURTLE",img_slide)
                nt_counter = 0
            nt_counter = nt_counter + 1


        # check if the ids list is not empty
        # if no check is added the code will crash
        if np.all(ids != None):

            # estimate pose of each marker and return the values
            # rvet and tvec-different from camera coefficients
            rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
            #(rvec-tvec).any() # get rid of that nasty numpy value array error

            for i in range(0, ids.size):
                # draw axis for the aruco markers
                aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)

            # draw a square around the markers
            aruco.drawDetectedMarkers(frame, corners)

            # code to show ids of the marker found
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0])+', '

            cv2.putText(frame, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)
            # cv2.putText(frame, "X:" + x_axis[], ())

        else:
            # code to show 'No Ids' when no markers are found
            cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

        # Horizontal Axis
        cv2.line(frame,(0, int(height/2)),(int(width),int(height/2)),(255,0,0),5)

        # Vertical Axis
        cv2.line(frame,(int(width/2),0),(int(width/2),int(height)),(255,0,0),5)
        
        # Line joining centre and marker
        # if(CP_X != 0 and CP_Y != 0 ):
        #     cv2.line(frame,(int(CP_X),int(CP_Y)),(int(width/2),int(height/2)),(255,0,0),5)

        cv2.imshow('frame',frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("Quit")
            break
        if key & 0xFF == ord('a'):
            print("Next Slide")
            previous_slide = True
            next_slide = False
            first_time = True
        if key & 0xFF == ord('d'):
            print("Previous Slide")
            previous_slide = False
            next_slide = True
            first_time = True
        if key & 0xFF == ord('h'):
            help_mode = not help_mode
            if(help_mode):
                print("Help activated")
            else:
                print("Help Deactivate")
        if key & 0xFF == ord('r'):
            print("Refresh")
            refresh = True
        if key & 0xFF == ord('f'):
            with open("output.pdf", "wb") as f:
                f.write(img2pdf.convert([i for i in slides if i.endswith(".jpg")]))
            break

# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()


# References
# 1. https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
# 2. https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
# 3. https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html