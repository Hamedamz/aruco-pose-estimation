'''
Sample Usage:-
python pi_pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100

python pi_pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --marker ARUCO --dict DICT_6X6_250 --camera 1
python pi_pose_estimation.py --K_Matrix calibration_matrix_hq_6mm.npy --D_Coeff distortion_coefficients_hq_6mm.npy --marker ARUCO --dict DICT_6X6_250 --camera 0

python pi_pose_estimation.py --K_Matrix calibration_matrix_hq_6mm.npy --D_Coeff distortion_coefficients_hq_6mm.npy --marker STAG --dict 21 --camera 0

python pi_pose_estimation.py --K_Matrix calibration_matrix_hq_6mm.npy --D_Coeff distortion_coefficients_hq_6mm.npy --marker APRILTAG --dict tag36h11 --camera 0


'''


import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
from picamera2 import Picamera2
# import stag
from enum import Enum
# import apriltag
import math
import os
import json
from timeit import default_timer as timer
from datetime import datetime
import logging


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

camera_map = {
    "pi3": 1,
    "pi3w": 1,
    "pihq6mm": 0,
}


res_map = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
}

class Marker(Enum):
    ARUCO = 1
    STAG = 2
    APRILTAG = 3
    

def yawpitchrolldecomposition(R):
    sin_x    = np.sqrt(R[2,0] * R[2,0] +  R[2,1] * R[2,1])    
    validity  = sin_x < 1e-6
    if not validity:
        z1    = math.atan2(R[2,0], R[2,1])     # around z1-axis
        x      = math.atan2(sin_x,  R[2,2])     # around x-axis
        z2    = math.atan2(R[0,2], -R[1,2])    # around z2-axis
    else: # gimbal lock
        z1    = 0                                         # around z1-axis
        x      = math.atan2(sin_x,  R[2,2])     # around x-axis
        z2    = 0                                         # around z2-axis

    yawpitchroll_angles = np.array([[z1], [x], [z2]])

    yawpitchroll_angles = -180*yawpitchroll_angles/math.pi
    yawpitchroll_angles[0,0] = (360-yawpitchroll_angles[0,0])%360 # change rotation sense if needed, comment this line otherwise
    yawpitchroll_angles[1,0] = yawpitchroll_angles[1,0]+90
    return yawpitchroll_angles


def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []

    #for i, c in enumerate(corners):
    nada, R, t = cv2.solvePnP(marker_points, corners, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
    rvecs.append(R)
    tvecs.append(t)
    trash.append(nada)
    
    rmat, jacobian = cv2.Rodrigues(R)
    camera_position = -np.matrix(rmat).T * np.matrix(t)
    camera_orientation = yawpitchrolldecomposition(rmat)
#         print("marker location", t)
    #print(f"camera location {i}", camera_position)
    #print(f"camera orientation {i}", camera_orientation)
#         camera_position = -rvec.transpose() @ np.array(tvecs)

    # print(tvecs)
#     rvec, jacobian = cv2.Rodrigues(np.array(rvecs))
#     camera_position = -rvec.transpose() @ np.array(tvecs)
#     print("marker location", tvecs)
#     print("camera location", camera_position)
    rvecs = cv2.UMat(np.array(rvecs))
    tvecs = cv2.UMat(np.array(tvecs))

    return rvecs, tvecs, trash, camera_position, camera_orientation


def pose_esitmation(frame, matrix_coefficients, distortion_coefficients, marker_type, dict_type):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it


    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pos = []
    ori = []
    if marker_type is Marker.ARUCO:
        aruco_dict_type = ARUCO_DICT[dict_type]
        arucoDict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
        corners, ids, rejected_corners = detector.detectMarkers(gray)

    elif marker_type is Marker.STAG:
        corners, ids, rejected_corners = stag.detectMarkers(frame, int(dict_type))
        
    elif marker_type is Marker.APRILTAG:
        april_options = apriltag.DetectorOptions(families=dict_type)
        april_detector = apriltag.Detector(april_options)
        results = april_detector.detect(gray)
        corners = []
        ids = []
        for r in results:
            corners.append(np.array([r.corners], dtype='float32'))
            ids.append([r.tag_id])
        ids = np.array(ids)

#     print(corners)
    
    if len(corners) > 0:
        for i in range(0, len(ids)):
            refined_corners = cv2.cornerSubPix(gray, corners[i], (11, 11), (-1, -1), criteria)
            nada, rvec, tvec = cv2.solvePnP(marker_points, refined_corners, matrix_coefficients, distortion_coefficients, False, cv2.SOLVEPNP_IPPE_SQUARE)

            rmat, jacobian = cv2.Rodrigues(rvec)
            # camera position
            pos.append(-np.matrix(rmat).T * np.matrix(tvec))
            # camera orientation
            ori.append(yawpitchrolldecomposition(rmat))

            if args["live"]:
                gray = cv2.drawFrameAxes(gray, matrix_coefficients, distortion_coefficients, rvec, tvec, length=0.01)
        
        if args["live"]:
            gray = cv2.aruco.drawDetectedMarkers(gray, corners, ids)

    if ids is None:
        ids = []
    else:
        ids = ids.tolist()           
    return gray, ids, pos, ori


def save_data(images, data):
    results_dir = datetime.now().strftime("%H%M%S_%m%d%Y")
    images_dir = os.path.join(results_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "data.json"), "w") as f:
        json.dump(data, f)
    
    for t, im in zip(data["timestamp"], images):
        cv2.imwrite(os.path.join(images_dir, f"{t}.jpg"), im)
        


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--camera", required=True, type=str, help="One of pi3, pi3w, or pihq6mm")
    ap.add_argument("-s", "--marker_size", type=float, default=0.02, help="Dimention of marker (meter)")
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-m", "--marker", type=str, default="ARUCO", help="Type of tag to detect. One of ARUCO, APRILTAG, or STAG")
    ap.add_argument("-c", "--dict", type=str, default="DICT_4X4_100", help="Type of dictionary of tag to detect")
    ap.add_argument("-t", "--duration", type=int, default=60, help="Duration of sampling (second)")
    ap.add_argument("-n", "--sample", type=str, default=30, help="Number of samples per second")
    ap.add_argument("-w", "--width", type=int, default=640, help="Width of image")
    ap.add_argument("-y", "--height", type=int, default=480, help="Height of image")
    ap.add_argument("-v", "--live", type=bool, default=False, help="Show live camera image")
    ap.add_argument("-r", "--res", type=str, help="Image resolution, one of 480p, 720p, 1080p, or 1440p, overwrites width and height")
    ap.add_argument("-g", "--debug", type=bool, default=False, help="Print logs")
    ap.add_argument("-o", "--save", type=bool, default=False, help="Save data")
    args = vars(ap.parse_args())

    data = {
        "timestamp": [],
        "ids": [],
        "duration": [],
        "position": [],
        "orientation": [],
        "args": args,
    }
    
    images = []
    
    if args["debug"]:
        logging.getLogger().setLevel(logging.INFO)
    image_size = (args["width"], args["height"])
    if args["res"] is not None:
        image_size = res_map[args["res"]]
    marker_size = args["marker_size"]
    dict_type = args["dict"]
    marker_type = Marker[args["marker"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)
    
    if args["live"]:
        cv2.startWindowThread()

    picam2 = Picamera2(camera_map[args["camera"]])
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": image_size}))
    picam2.start()
    time.sleep(2)

    # capture frames from the camera
    while True:
        start = timer()
        im = picam2.capture_array()
        output, ids, pos, ori = pose_esitmation(im, k, d, marker_type, dict_type)
        end = timer()

        data["timestamp"].append(start)
        data["duration"].append(end - start)
        data["position"].append([p.tolist() for p in pos])
        data["orientation"].append([o.tolist() for o in ori])
        data["ids"].append(ids)
        images.append(im)
        
        logging.info(f"ids:\n{ids}\n\nposisions:\n{pos}\n\norientations:\n{ori}\n\n")
        
        if args["live"]:
            cv2.imshow('Estimated Pose', output)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("i"):
            marker_size = float(input("input marker size in mm:"))/1000
        elif key == ord("1"):
            marker_size = 0.02
        elif key == ord("2"):
            marker_size = 0.015
        elif key == ord("3"):
            marker_size = 0.01
        elif key == ord("4"):
            marker_size = 0.005
        elif key == ord("5"):
            marker_size = 0.0025

    if args["live"]:
        cv2.destroyAllWindows()

    if args["save"]:
        save_data(images, data)

