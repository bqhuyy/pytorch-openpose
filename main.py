import sys
sys.path.insert(0, 'python')
import cv2
import model
import util
from hand import Hand
from body import Body
import matplotlib.pyplot as plt
import copy
import numpy as np
import re
import os

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

def draw_pose(oriImg):
  candidate, subset = body_estimation(oriImg)
  canvas = copy.deepcopy(oriImg)
  canvas = util.draw_bodypose(canvas, candidate, subset)
  # detect hand
  hands_list = util.handDetect(candidate, subset, oriImg)
  all_hand_peaks = []
  for x, y, w, is_left in hands_list:
    peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
    peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
    peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
    all_hand_peaks.append(peaks)

  canvas = util.draw_handpose(canvas, all_hand_peaks)
  return canvas

def numericalSort(value):
  numbers = re.compile(r'(\d+)')
  parts = numbers.split(value)
  parts[1::2] = map(int, parts[1::2])
  return parts

def create_video(video_path, video_name):     
  img_array = []
  filename = sorted(os.listdir(video_path), key=numericalSort)
  for i in filename:
    f = os.path.join(video_path, i)
    print(f)
    oriImg = cv2.imread(f)
    oriImg = draw_pose(oriImg)
    img_array.append(oriImg)
    
  height, width, layers = img_array[0].shape
  size = (width, height)
  output = cv2.VideoWriter('./result/'+video_name+'.avi',cv2.VideoWriter_fourcc(*'XVID'), 10, size)

  for i in range(len(img_array)):
    output.write(img_array[i])
  output.release()

path = './samples/GT_frames/'
for i in os.listdir(path):
  folder = os.path.join(path,i)
  print(folder)
  create_video(folder, i)
