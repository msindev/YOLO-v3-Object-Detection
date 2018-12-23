import numpy as np
import cv2

confidenceThreshold = 0.5
NMSThreshold = 0.3

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'
labels = open(labelsPath).read().strip().split('\n')

np.random.seed(10)
COLORS = np.random.randint(0, 255, size = (len(labels), 3), dtype = 'unit8')
