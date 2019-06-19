import numpy
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import test_simple
import cv2
import video_converter
import math
from datetime import datetime

from imageai.Detection import ObjectDetection
import os

def drawBounds(x1,y1,x2,y2,picture):
    for i in range(x1,x2 - 1):
        picture[0][0][y1][i] = 0
        picture[0][0][y2-1][i] = 0

    for i in range(y1,y2-1):
        picture[0][0][i][x1] = 0
        picture[0][0][i][x2-1] = 0

def drawBoundsColor(x1,y1,x2,y2,picture):
    for i in range(x1,x2):
        picture[y1][i][0] = 0
        picture[y2][i][0] = 0

    for i in range(y1,y2):
        picture[i][x1][0] = 0
        picture[i][x2][0] = 0
def findAverage(x1,y1,x2,y2,picture):
    count = 1
    total = 0
    for i in range(x1,x2 - 1):
        for j in range(y1,y2 - 1):
            count += 1
            total += picture[0][0][j][i]
    return(total / count)

def getDistnace(input):
    return(25.4662 * math.pow(input,(-1.3284)))

#execution_path = os.getcwd()

pictureAmount = video_converter.findCount()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

execution_path = os.getcwd()
path = os.getcwd() + '\\assets\\videos\\'

count = 1



while(count < 10):
    current = datetime.now()
    pictureName = ("%d" % count)

    image = cv2.imread(path + pictureName + '.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path + '\\models' , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()

    test_simple.test_simple_ethan(path + pictureName + '.jpg','mono_640x192')
    print('Finished Monodepth2 ' + str(datetime.now() - current))
    current = datetime.now()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(path +  pictureName + ".jpg"), output_image_path=os.path.join(execution_path + '\\assets\\proccessed', pictureName + "new.jpg"))
    print('Object detecting ' + str(datetime.now() - current))
    current = datetime.now()
    nPic = numpy.load(path + pictureName + '_disp.npy')
    nPicFlat = numpy.ones((nPic.size,1))

    nPicReal = imread(path + pictureName + '.jpg')
    tempcount = 0
    max = 0
    for i in range((len(nPic[0][0]))):
        for j in range(len(nPic[0][0][0])):
            nPicFlat[tempcount] = nPic[0][0][i][j]
            if(max < nPicFlat[count]):
                max = nPicFlat[count]
            tempcount = tempcount + 1



    constantX = len(nPic[0][0][0])
    constantY = len(nPic[0][0])

    originalX = len(nPicReal[0])
    originalY = len(nPicReal)

    averages = []

    for (x,y,w,h) in faces:
        x1 = (int)(round(x * constantX / originalX))
        y1 = (int)(round(y * constantY / originalY))
        x2 = (int)(round(w * constantX / originalX))
        y2 = (int)(round(h * constantY / originalY))
        drawBounds(x1,y1,x2,y2,nPic)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
                x1 = (int)(round(ex * constantX / originalX))
                y1 = (int)(round(ey * constantY / originalY))
                x2 = (int)(round(ew * constantX / originalX))
                y2 = (int)(round(eh * constantY / originalY))
                drawBounds(x1,y1,x2,y2,nPic)

    for eachObject in detections:
        x1 = round(eachObject['box_points'][0] * constantX / originalX)
        x2 = round(eachObject['box_points'][2] * constantX / originalX)
        y1 = round(eachObject['box_points'][1] * constantY / originalY)
        y2 = round(eachObject['box_points'][3] * constantY / originalY)
        eachObject['box_points'] = [x1,y1,x2,y2]

    tempcount = 0
    for eachObject in detections:
        x1 = (int)(eachObject['box_points'][0])
        y1 = (int)(eachObject['box_points'][1])
        x2 = (int)(eachObject['box_points'][2])
        y2 = (int)(eachObject['box_points'][3])
        averages.append(getDistnace(findAverage(x1,y1,x2,y2,nPic)))
        drawBounds((x1),(y1),(x2),(y2),nPic)
        tempcount += 1

    for i in range(len(averages)):
        print(detections[i]['name'] + str(averages[i]))
    mat = nPic[0][0] / max

    img = Image.fromarray(numpy.uint8(mat * 255) , 'L')
    img.save("assets//proccessed//Proccessedframe%d.jpg" % count)
    count += 1
    #cv2.imwrite("assets//videos//Proccessedframe%d.jpg" % pictureAmount, threshed)
    #img.show()
    #plt.show()
