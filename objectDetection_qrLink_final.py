# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""

import argparse
import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import qrcode
from picamera2 import Picamera2, Preview
import time
import cv2
import cameraTest


def check_for_dark():
  img = cv2.imread("html/rgb.jpg")
  number_of_black_pix = np.sum(img == 0)
  print('Number of black pixels: ', number_of_black_pix)


def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def first_word(word):
  word_first = word.split(',')
  w = word_first[0]
  w= w[0].upper()+w[1:len(w)]
  w=w.replace(" ", "_")
  return w

def object(word):
  ans=""
  for i in word:
    if i != ',':
      ans=ans+i

def run_object_detection():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='./html/rgb.jpg',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='./mobilenet/mobilenet_v1_1.0_224.tflite',
      help='.tflite model to be executed')
  parser.add_argument(
      '-l',
      '--label_file',
      default='./mobilenet/labels.txt',
      help='name of file containing labels')
  parser.add_argument(
      '--input_mean',
      default=127.5, type=float,
      help='input_mean')
  parser.add_argument(
      '--input_std',
      default=127.5, type=float,
      help='input standard deviation')
  parser.add_argument(
      '--num_threads', default=None, type=int, help='number of threads')
  parser.add_argument(
      '-e', '--ext_delegate', help='external_delegate_library path')
  parser.add_argument(
      '-o',
      '--ext_delegate_options',
      help='external delegate options, \
            format: "option1: value1; option2: value2"')

  args = parser.parse_args()

  ext_delegate = None
  ext_delegate_options = {}

  # parse extenal delegate options
  if args.ext_delegate_options is not None:
    options = args.ext_delegate_options.split(';')
    for o in options:
      kv = o.split(':')
      if (len(kv) == 2):
        ext_delegate_options[kv[0].strip()] = kv[1].strip()
      else:
        raise RuntimeError('Error parsing delegate option: ' + o)

  # load external delegate
  if args.ext_delegate is not None:
    print('Loading external delegate from {} with args: {}'.format(
        args.ext_delegate, ext_delegate_options))
    ext_delegate = [
        tflite.load_delegate(args.ext_delegate, ext_delegate_options)
    ]

  interpreter = tflite.Interpreter(
      model_path=args.model_file,
      experimental_delegates=ext_delegate,
      num_threads=args.num_threads)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(args.image).resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - args.input_mean) / args.input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(args.label_file)
  # for i in top_k:
  #   if floating_model:
  #     print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
  #   else:
  #     print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
  print()
  print()
  print("Most likely object: "+labels[top_k[0]][4:len(labels[top_k[0]])])
  print()
  text=labels[top_k[0]][4:len(labels[top_k[0]])]
  f = open("html/objectFile.txt", "a")
  f.truncate(0)
  f.write(text)
  f.close()
  img = qrcode.make("https://en.wikipedia.org/wiki/"+first_word(labels[top_k[0]][4:len(labels[top_k[0]])]))
  type(img)  
  img.save("html/QR_code.png")


  # print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
  print()
  str = input("Would you like to learn more about this object?[Yes/No]:  ")
  if str == "Yes":
    print()
    print("Sending Qr code for top result...")
    print()
    
    print("Done! Look at 'QR_code.png' to get QR Code.")
    print()
  else:
    print()
    print("Thank you!")

  

