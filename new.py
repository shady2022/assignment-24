import cv2
from imutils.face_utils.helpers import FACIAL_LANDMARKS_IDXS
import numpy as np
import dlib
from imutils import face_utils
from collections import OrderedDict

FACIAL_LANDMARKS_IDXS = OrderedDict([("mouth", (48, 68)),
                                ("left_eye", (22, 27)),
                                ("right_eye",(36, 42))])

def visualize_facial_landmarks(image, shape, colors = None, alpha= 0.75):
    overlay = image.shape
    output = image.copy
    if colors is None:
        colors = [(19, 199, 109),(158, 163, 32), (168, 100, 168)]
        
        for (i,name) in enumerate(FACIAL_LANDMARKS_IDXS()):
            (l, r) = FACIAL_LANDMARKS_IDXS[name]
            pts = shape [l : r]
            
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)
            
        cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
    return output

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dot")

image = cv2.imread("myface.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

for (i,rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
    
    output = visualize_facial_landmarks (image, shape)
    cv2.imshow("image", output)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()


