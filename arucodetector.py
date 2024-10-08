import cv2 as cv
import numpy as np
import time
import sys

ARUCO_DICT = {
    "ARUCO_4X4_1000": cv.aruco.DICT_4X4_1000,
    "ARUCO_5X5_250": cv.aruco.DICT_5X5_250
    }

def arucodraw(corners, ids,rejected, image):
    if len(corners) > 0:

        ids = ids.flatten()

        for(markercorner, markerid) in zip(corners, ids):
            corners = markercorner.reshape((4,2))
            (tl, tr, br, bl) = corners

            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            tl = (int(tl[0]), int(tl[1]))

            cv.line(image, tr, br, (0,255,0), 2)
            cv.line(image, tl, bl, (0,255,0), 2)
            cv.line(image, tr, tl, (0,255,0), 2)
            cv.line(image, br, bl, (0,255,0), 2)

            cx = int((tl[0] + br[0]) / 2.0)
            cy = int((tl[1] + br[1]) / 2.0)

            cv.circle(image, (cx, cy), 50, (0, 0, 255), -1)

    return image


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv.aruco.DetectorParameters()

    corners, ids, rejected_imgpoints = cv.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if len(corners) > 0:
        for i in range(0, len(ids)): 
            rvec, tvec, markerpoints = cv.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)

            cv.aruco.drawDetectedMarkers(frame, corners)

            cv.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

            distance = np.linalg.norm(tvec[i][0])

            cv.putText(frame, f"Distance: {distance*100} meters", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    return frame

aruco_type = "ARUCO_5X5_250"
id = 3

arucoparams = cv.aruco.DetectorParameters()

arucoDict = cv.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

intrinsic_camera = np.array(((758.37212254, 0, 311.18706386),(0, 760.2837291, 238.39821263),(0,0, 1)))
distortion = np.array((0.18532645, -0.6230805, -0.0087438, -0.00252058, 1.73210502))

# print("Aruco type {} with ID {}".format(aruco_type, id))
# tag_size = 250
# tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
# cv.aruco.generateImageMarker(arucodict, id, tag_size, tag, 1)

# tag_name = "arucomarker/" + aruco_type + '_' + str(id) + ".png"
# cv.imwrite(tag_name, tag)
# cv.imshow("arucomarker", tag)

# cv.waitKey(0)

cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, img = cap.read()

    h,w,_ = img.shape
    width = 1000
    height = int(width*(h/w))
    img = cv.resize(img, (width, height), interpolation = cv.INTER_CUBIC)

    corners, ids, rejected = cv.aruco.detectMarkers(img, arucoDict, parameters=arucoparams)

    # detected_markers = arucodraw(corners, ids, rejected, img)

    output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)
    cv.imshow('imgae', output)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
