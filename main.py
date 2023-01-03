import cv2
import os 
import pyopenpose as op
import numpy as np

image_path = "testimage.jpg"

params = dict()
params["model_folder"] = "D:\CodeHome\openpose\models"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    show_frame = np.concatenate((frame, datum.cvOutputData), axis=1)
    print(datum.poseKeypoints.shape)
    print("----------------------------------")

    cv2.imshow('img', show_frame)
    # exit()
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
