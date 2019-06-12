import numpy
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import test_simple

from imageai.Detection import ObjectDetection
import os

def drawBounds(x1,y1,x2,y2,picture):
    for i in range(x1,x2):
        picture[0][0][y1][i] = 0
        picture[0][0][y2][i] = 0

    for i in range(y1,y2):
        picture[0][0][i][x1] = 0
        picture[0][0][i][x2] = 0

def findAverage(x1,y1,x2,y2,picture):
    count = 1
    total = 0
    for i in range(x1,x2):
        for j in range(y1,y2):
            count += 1
            total += picture[0][0][j][i]
    return(total / count)


execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path + '\\models' , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

pictureName = 'idiot1'

test_simple.test_simple_ethan('assets\\' + pictureName + '.jpg','mono_1024x320')

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path + '\\assets', pictureName + ".jpg"), output_image_path=os.path.join(execution_path + '\\assets', pictureName + "new.jpg"))
path = os.getcwd() + '\\assets\\'

nPic = numpy.load(path + pictureName + '_disp.npy')
nPicFlat = numpy.ones((nPic.size,1))

nPicReal = imread(path + pictureName + '.jpg')
count = 0
max = 0
for i in range((len(nPic[0][0]))):
    for j in range(len(nPic[0][0][0])):
        nPicFlat[count] = nPic[0][0][i][j]
        if(max < nPicFlat[count]):
            max = nPicFlat[count]
        count = count + 1



constantX = len(nPic[0][0][0])
constantY = len(nPic[0][0])

originalX = len(nPicReal[0])
originalY = len(nPicReal)

averages = []


for eachObject in detections:
    x1 = round(eachObject['box_points'][0] * constantX / originalX)
    x2 = round(eachObject['box_points'][2] * constantX / originalX)
    y1 = round(eachObject['box_points'][1] * constantY / originalY)
    y2 = round(eachObject['box_points'][3] * constantY / originalY)
    eachObject['box_points'] = [x1,y1,x2,y2]
    print(eachObject['box_points'])
count = 0
for eachObject in detections:
    x1 = (int)(eachObject['box_points'][0])
    y1 = (int)(eachObject['box_points'][1])
    x2 = (int)(eachObject['box_points'][2])
    y2 = (int)(eachObject['box_points'][3])
    averages.append(findAverage(x1,y1,x2,y2,nPic))
    drawBounds((x1),(y1),(x2),(y2),nPic)
    count += 1


plt.hist(nPicFlat)
for i in range(len(averages)):
    print(detections[i]['name'] + str(averages[i]))
mat = nPic[0][0] / max
img = Image.fromarray(numpy.uint8(mat * 255) , 'L')
img.show()
plt.show()
