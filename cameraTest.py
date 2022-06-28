from picamera2 import Picamera2, Preview
import time
import cv2
import numpy as np


picam2 = Picamera2()
config = picam2.preview_configuration(main={"size": (640, 480)},
                                    lores={"size": (320, 240), "format": "YUV420"})
picam2.configure(config)
(w0, h0) = picam2.stream_configuration("main")["size"]
(w1, h1) = picam2.stream_configuration("lores")["size"]
s1 = picam2.stream_configuration("lores")["stride"]


def take_picture():
    picam2.start()
    time.sleep(1)
    yuv420 = picam2.capture_array("lores")
    time.sleep(1)
    rgb = cv2.cvtColor(yuv420, cv2.COLOR_YUV420p2RGB)
    cv2.imwrite("rgb.jpg", rgb)
    cv2.destroyAllWindows()
    picam2.stop()



