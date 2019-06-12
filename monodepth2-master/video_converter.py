import cv2
import os
def convertVideoToPictures(video_name):
    if not os.path.exists('assets//videos'):
        os.mkdir('assets//videos')
    vidcap = cv2.VideoCapture('assets//videos//' + video_name)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite("assets//videos//frame%d.jpg" % count, image)     # save frame as JPEG file
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1
