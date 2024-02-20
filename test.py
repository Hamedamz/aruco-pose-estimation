from picamera2 import Picamera2, Preview
import time
image_size = (4608, 2592)
picam2 = Picamera2(1)
camera_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": image_size})
picam2.configure(camera_config)
picam2.start_preview(Preview.QTGL)
picam2.start()
time.sleep(5)
picam2.capture_file("test.jpg")

#!/usr/bin/python3

# import cv2
# 
# from picamera2 import Picamera2
# 
# # Grab images as numpy arrays and leave everything else to OpenCV.
# 
# face_detector = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
# cv2.startWindowThread()
# 
# picam2 = Picamera2(1)
# picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
# picam2.start()
# 
# while True:
#     im = picam2.capture_array()
# 
#     grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(grey, 1.1, 5)
# 
#     for (x, y, w, h) in faces:
#         cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0))
# 
#     cv2.imshow("Camera", im)
#     cv2.waitKey(1)