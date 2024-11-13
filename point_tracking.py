'''
Sample Usage:-
python pi_pose_estimation.py -i pi3 -r 720p -e test -t 1000 --marker_size 0.02 --live
'''

import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
# import stag
# import apriltag
import math
import os
import json
from timeit import default_timer as timer
from datetime import datetime
import logging
from consts import res_map, camera_map, Marker
import scipy.spatial.transform as transform
from worker_socket import WorkerSocket
import socket
import struct

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def yawpitchrolldecomposition(R):
    rotation = transform.Rotation.from_matrix(R)
    euler_angles = rotation.as_euler('xyz', degrees=True)

    #     sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    #     singular = sy < 1e-6
    #
    #     if not singular:
    #         x = np.arctan2(R[2,1], R[2,2])
    #         y= np.arctan2(-R[2,0], sy)
    #         z= np.arctan2(R[1,0], R[0,0])
    #     else:
    #         x = np.arctan2(-R[1,2], R[1,1])
    #         y = np.arctan2(-R[2,0], sy)
    #         z = 0

    #     sin_x    = np.sqrt(R[2,0] * R[2,0] +  R[2,1] * R[2,1])
    #     validity  = sin_x < 1e-6
    #     if not validity:
    #         z1    = math.atan2(R[2,0], R[2,1])     # around z1-axis
    #         x      = math.atan2(sin_x,  R[2,2])     # around x-axis
    #         z2    = math.atan2(R[0,2], -R[1,2])    # around z2-axis
    #     else: # gimbal lock
    #         z1    = 0                                         # around z1-axis
    #         x      = math.atan2(sin_x,  R[2,2])     # around x-axis
    #         z2    = 0                                         # around z2-axis

    #     yawpitchroll_angles = np.array([[z1], [x], [z2]])

    #     yawpitchroll_angles = np.array([[x], [y], [z]])
    #     yawpitchroll_angles = -180*yawpitchroll_angles/math.pi
    #     yawpitchroll_angles[0,0] = (360-yawpitchroll_angles[0,0])%360 # change rotation sense if needed, comment this line otherwise
    #     yawpitchroll_angles[1,0] = yawpitchroll_angles[1,0]+90
    return euler_angles


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

    # for i, c in enumerate(corners):
    nada, R, t = cv2.solvePnP(marker_points, corners, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
    rvecs.append(R)
    tvecs.append(t)
    trash.append(nada)

    rmat, jacobian = cv2.Rodrigues(R)
    camera_position = -np.matrix(rmat) * np.matrix(t)
    camera_orientation = yawpitchrolldecomposition(rmat)
    #         print("marker location", t)
    # print(f"camera location {i}", camera_position)
    # print(f"camera orientation {i}", camera_orientation)
    #         camera_position = -rvec.transpose() @ np.array(tvecs)

    # print(tvecs)
    #     rvec, jacobian = cv2.Rodrigues(np.array(rvecs))
    #     camera_position = -rvec.transpose() @ np.array(tvecs)
    #     print("marker location", tvecs)
    #     print("camera location", camera_position)
    rvecs = cv2.UMat(np.array(rvecs))
    tvecs = cv2.UMat(np.array(tvecs))

    return rvecs, tvecs, trash, camera_position, camera_orientation


def pose_estimation(frame, matrix_coefficients, distortion_coefficients, marker_type, dict_type):
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
            nada, rvec, tvec = cv2.solvePnP(marker_points, refined_corners, matrix_coefficients,
                                            distortion_coefficients, False, cv2.SOLVEPNP_IPPE_SQUARE)

            rmat, jacobian = cv2.Rodrigues(rvec)
            # camera position
            # pos.append(-np.matrix(rmat).T * np.matrix(tvec))
            # marker position
            pos.append(tvec)
            # marker orientation
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


def count_empty_lists(lst):
    count = 0
    for item in lst:
        if not item:
            count += 1

    return count


def save_data(images, data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"{timestamp}_{args['camera']}_{args['res']}")
    images_dir = os.path.join(results_dir, 'images')

    for i in range(len(data["ids"])):
        data["ids"][i] = [j[0] for j in data["ids"][i]]

    for i in range(len(data["position"])):
        for j in range(len(data["position"][i])):
            data["position"][i][j] = [p[0] for p in data["position"][i][j].tolist()]

    for i in range(len(data["orientation"])):
        for j in range(len(data["orientation"][i])):
            data["orientation"][i][j] = data["orientation"][i][j].tolist()

    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)

    total_ims = len(data["timestamp"])
    exp_dur = data["timestamp"][-1] - data["timestamp"][0]
    filtered_pos = list(filter(bool, data["position"]))
    filtered_ori = list(filter(bool, data["orientation"]))
    stats = {
        "exp_duration": exp_dur,
        "total_images": total_ims,
        "image_per_second": total_ims / exp_dur,
        "avg_camera_delay": sum(data["camera_delay"]) / total_ims,
        "avg_alg_delay": sum(data["alg_delay"]) / total_ims,
        "avg_position": np.mean(filtered_pos, axis=0).tolist(),
        "avg_distance": np.mean(np.linalg.norm(filtered_pos, axis=2), axis=0).tolist(),
        "avg_abs_orientation": np.mean(np.abs(filtered_ori), axis=0).tolist(),
        "detection_rate": 1 - count_empty_lists(data["ids"]) / total_ims,
    }

    data["stats"] = stats

    with open(os.path.join(results_dir, "data.json"), "w") as f:
        json.dump(data, f)

    for t, im in zip(data["timestamp"], images):
        cv2.imwrite(os.path.join(images_dir, f"{t}.jpg"), im)


def find_dot(img):
    # img = cv.GaussianBlur(img,(5,5),0)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grey = cv2.threshold(grey, 255 * 0.2, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    image_points = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            cv2.putText(img, f'({center_x}, {center_y})', (center_x, center_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (100, 255, 100), 1)
            cv2.circle(img, (center_x, center_y), 1, (100, 255, 100), -1)
            image_points.append([center_x, center_y])

    if len(image_points) == 0:
        image_points = [[None, None]]

    return img, image_points


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--camera", required=True, type=str, help="One of aideck, pi3, pi3w, or pihq6mm")
    ap.add_argument("-s", "--marker_size", type=float, default=0.02, help="Dimention of marker (meter)")
    ap.add_argument("-k", "--K_Matrix", help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-m", "--marker", type=str, default="ARUCO",
                    help="Type of tag to detect. One of ARUCO, APRILTAG, or STAG")
    ap.add_argument("-c", "--dict", type=str, default="DICT_4X4_100", help="Type of dictionary of tag to detect")
    ap.add_argument("-t", "--duration", type=int, default=60, help="Duration of sampling (second)")
    ap.add_argument("-n", "--sample", type=str, default=30, help="Number of samples per second")
    ap.add_argument("-w", "--width", type=int, default=640, help="Width of image")
    ap.add_argument("-y", "--height", type=int, default=480, help="Height of image")
    ap.add_argument("-v", "--live", action="store_true", help="Show live camera image")
    ap.add_argument("-r", "--res", type=str, default="480p",
                    help="Image resolution, one of 244p, 480p, 720p, 1080p, or 1440p, overwrites width and height")
    ap.add_argument("-g", "--debug", action="store_true", help="Print logs")
    ap.add_argument("-o", "--save", action="store_true", help="Save data")
    ap.add_argument("-sm", "--sensor_modes", action="store_true", help="Print sensor modes of the camera and terminate")
    ap.add_argument("-b", "--broadcast", action="store_true",
                    help="Broadcast measurements modify worker socket with proper ip and port")
    ap.add_argument("-mr", "--max_fps", action="store_true", help="Use maximum fps")
    ap.add_argument("-e", "--note", type=str, help="Notes")
    ap.add_argument("-l", "--lenspos", type=float, help="Lens position for manual focus")
    args = vars(ap.parse_args())

    data = {
        "stats": {},
        "args": args,
        "timestamp": [],
        "ids": [],
        "alg_delay": [],
        "camera_delay": [],
        "position": [],
        "orientation": [],
    }

    images = []

    if args["broadcast"]:
        sock = WorkerSocket()

    if args["debug"]:
        logging.getLogger().setLevel(logging.INFO)
    image_size = (args["width"], args["height"])
    if args["res"] is not None:
        image_size = res_map[args["res"]]
    marker_size = args["marker_size"]
    dict_type = args["dict"]
    marker_type = Marker[args["marker"]]
    if args["K_Matrix"] is None:
        calibration_matrix_path = os.path.join("calibration", args["camera"], args["res"], "mtx.npy")
    else:
        calibration_matrix_path = args["K_Matrix"]
    if args["D_Coeff"] is None:
        distortion_coefficients_path = os.path.join("calibration", args["camera"], args["res"], "dist.npy")
    else:
        distortion_coefficients_path = args["D_Coeff"]
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    if args["live"]:
        cv2.startWindowThread()

    if args["camera"] == "aideck":
        deck_port = 5000
        deck_ip = "192.168.4.1"

        print("Connecting to socket on {}:{}...".format(deck_ip, deck_port))
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((deck_ip, deck_port))
        print("Socket connected")

        imgdata = None
        data_buffer = bytearray()


        def rx_bytes(size):
            data = bytearray()
            while len(data) < size:
                data.extend(client_socket.recv(size - len(data)))
            return data
    else:
        from picamera2 import Picamera2
        from libcamera import controls

        picam2 = Picamera2(camera_map[args["camera"]])
        modes = picam2.sensor_modes

        if args["sensor_modes"]:
            print(modes)
            exit()
        #     picam2.configure(picam2.create_preview_configuration(sensor={"output_size": mode["size"], "bit_depth": mode["bit_depth"]}, main={"size": image_size}))
        # cam_conf = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": image_size})
        cam_conf = picam2.create_video_configuration(main={"format": "XRGB8888", "size": image_size})
        picam2.configure(cam_conf)
        if args["max_fps"]:
            picam2.set_controls({"FrameDurationLimits": (8333, 8333)})

        if args["camera"] != "pihq6mm":
            if args["lenspos"]:
                picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": args['lenspos']})
            else:
                picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        # picam2.controls.ExposureTime = 2000
        picam2.start()
        time.sleep(2)

    # capture frames from the camera
    exp_start = time.time()
    msg = 0
    while True:
        start = time.time()

        if args["camera"] == "aideck":
            packetInfoRaw = rx_bytes(4)
            [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
            imgHeader = rx_bytes(length - 2)
            [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)

            if magic == 0xBC:
                imgStream = bytearray()

                while len(imgStream) < size:
                    packetInfoRaw = rx_bytes(4)
                    [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
                    chunk = rx_bytes(length - 2)
                    imgStream.extend(chunk)

                if format == 0:
                    bayer_img = np.frombuffer(imgStream, dtype=np.uint8)
                    bayer_img.shape = (244, 324)
                    color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGRA)
                    im = color_img
        else:
            im = picam2.capture_array()

        mid = time.time()
        im = cv2.undistort(im, k, d)
        im = cv2.GaussianBlur(im, (5, 5), 0)
        # kernel = np.array([[-2, -1, -1, -1, -2],
        #                    [-1, 1, 3, 1, -1],
        #                    [-1, 3, 4, 3, -1],
        #                    [-1, 1, 3, 1, -1],
        #                    [-2, -1, -1, -1, -2]])
        # im = cv2.filter2D(im, -1, kernel)
        im = cv2.convertScaleAbs(im, alpha=1.5, beta=0)
        im, dots = find_dot(im)
        # print(dots)
        output, ids, pos, ori = pose_estimation(im, k, d, marker_type, dict_type)
        end = time.time()

        if args["broadcast"]:
            if len(pos) and len(ori):
                p = pos[0].tolist()
                msg = f"{p[0][0]},{p[1][0]},{p[2][0]},{ori[0][0]},{ori[0][1]},{ori[0][2]}"
                # print(f"x={p[0][0]:.3f},y={p[1][0]:.3f},z={p[2][0]:.3f},norm={np.linalg.norm(np.array([p[0][0], p[1][0], p[2][0]])):.3f},{ori[0][0]:.2f},{ori[0][1]:.2f},{ori[0][2]:.2f}")

            if msg:
                sock.broadcast(msg)

        data["timestamp"].append(start)
        data["alg_delay"].append(end - mid)
        data["camera_delay"].append(mid - start)
        data["position"].append(pos)
        data["orientation"].append(ori)
        data["ids"].append(ids)

        if args["save"]:
            images.append(im)

        logging.info(f"ids:\n{ids}\n\nposisions:\n{pos}\n\norientations:\n{ori}\n\n")

        if args["live"]:
            cv2.rectangle(output, (image_size[0] // 2 - 50, image_size[1] // 2 - 50),
                          (image_size[0] // 2 + 50, image_size[1] // 2 + 50), (0, 255, 0), 1)
            cv2.imshow('Estimated Pose', output)

        if end - exp_start >= args["duration"]:
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("i"):
            marker_size = float(input("input marker size in mm:")) / 1000
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

    lens_position = picam2.capture_metadata()["LensPosition"]
    logging.info(f"lens position: {lens_position}")
