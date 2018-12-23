import numpy as np
import cv2

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'
labels = open(labelsPath).read().strip().split('\n')

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

image = 'images/dog.jpg'
(H, W) = image.shape[:2]

#Determine output layer names
layerName = net.getLayerNames()
layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
net.setInput(blob)
layersOutputs = net.forward(layerName)

boxes = []
confidences = []
classIDs = []
