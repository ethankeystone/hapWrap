import test_simple
import os
from imageai.Detection import ObjectDetection
path = os.getcwd() + '\\assets\\old\\'
pictureName = '7'


execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path + '\\models' , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=os.path.join(path +  pictureName + ".jpg"), output_image_path=os.path.join(execution_path + '\\assets\\proccessed', pictureName + "new.jpg"))

for eachObject in detections:
    x1 = round(eachObject['box_points'][0])
    x2 = round(eachObject['box_points'][2])
    y1 = round(eachObject['box_points'][1])
    y2 = round(eachObject['box_points'][3])
    print(eachObject['name'])
    print(str(x2 - x1) + " = x | " + str(y2 - y1) + " = y")
