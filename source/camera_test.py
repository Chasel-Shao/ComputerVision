import cv2
import scipy
from scipy import signal
from matplotlib import pyplot as plt
import pylab
import math
import numpy as np
from convolution_function import *


cap = cv2.VideoCapture(0)
while(1):
    # get a frame
    ret, frame = cap.read()
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    v = np.median(blurred)
    lower = int(max(0, .5 * v))
    upper = int(min(255, 1.5 * v))
    output = cv2.Canny(blurred, lower, upper)
    # show a frame
    cv2.imshow("capture", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
