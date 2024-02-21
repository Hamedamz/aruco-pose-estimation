import cv2
from picamera2 import Picamera2, Preview
import time
import argparse
from consts import res_map, camera_map
import os


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--camera", type=str, required=True, help="One of pi3, pi3w, or pihq6mm")
    ap.add_argument("-o", "--output", type=str, help="Output path")
    ap.add_argument("-r", "--res", type=str, required=True, help="Image resolution, one of 480p, 720p, 1080p, or 1440p, overwrites width and height")
    args = vars(ap.parse_args())
    
    image_size = res_map[args["res"]]
    
    if args["output"] is None:
        images_dir = os.path.join("calibration", args["camera"], args["res"])
    else:
        images_dir = args["output"]
    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)

    picam2 = Picamera2(camera_map[args["camera"]])
    camera_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": image_size})
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)
    num = 0

    while True:
        img = picam2.capture_array()

        k = cv2.waitKey(5)

        if k == 27:
            break
        elif k == ord('s'): # wait for 's' key to save and exit
            cv2.imwrite(os.path.join(images_dir, f"{str(num)}.jpg"), img)
            print("image saved!")
            num += 1

        cv2.imshow('Img',img)

    # Release and destroy all windows before termination
    cv2.destroyAllWindows()
