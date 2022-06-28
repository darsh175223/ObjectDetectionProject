import cameraTest
import cv2
import numpy as np
import time
import objectDetection_qrLink_final as detection
imagePath="html/rgb.jpg"
def check_for_dark():
  img = cv2.imread(imagePath)
  number_of_black_pix = np.sum(img == 0)
  return (number_of_black_pix>9000)

def wait_for_dark():
  cameraTest.take_picture()
  while(True):
    cameraTest.take_picture()
    print("Cover the camera to take a picture!")
    print()
    if check_for_dark():
      break
  print("It's dark")

def first_word(word):
  ans=""
  for i in word:
    if i != ',':
      ans=ans+i

if __name__ == '__main__':
  wait_for_dark()
  time.sleep(3)
  print("Taking photo in: ")
  print("3...")
  time.sleep(2)
  print("2...")
  time.sleep(2)
  print("1...")
  time.sleep(2)
  cameraTest.take_picture()
  print("picture taken")
  detection.run_object_detection()
