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

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

video_path = "./demo.mov"
video = cv2.VideoCapture(video_path)
frame = 0
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# frame
print('Total frames: ', length)

img_array = []

while (True):
    ret, oriImg = video.read() # B,G,R order
    print(frame)
    if not ret:
        break
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
    img_array.append(canvas)
    frame += 1
    
height, width, layers = img_array[0].shape
size = (width, height)
output = cv2.VideoWriter('demo.avi',cv2.VideoWriter_fourcc(*'XVID'), 30, size)

for i in range(len(img_array)):
    output.write(img_array[i])
output.release()
