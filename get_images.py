import cv2
from picamera2 import Picamera2, Preview
import time
import argparse



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--camera", required=True, help="Set to True if using webcam")
    args = vars(ap.parse_args())
    
    image_size = (4608//2, 2592//2)
#     image_size = (640, 480)
    picam2 = Picamera2(int(args["camera"]))
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
            cv2.imwrite('img/wide/2/' + str(num) + '.png', img)
            print("image saved!")
            num += 1

        cv2.imshow('Img',img)

    # Release and destroy all windows before termination

    cv2.destroyAllWindows()
