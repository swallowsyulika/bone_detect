import cv2
import os 
import pyopenpose as op
import numpy as np
import socket
import cv2
import pickle
import struct ## new
import time
from pathlib import Path

record_mode = False
save_count = 0

checkpoint_root = Path("dataset")
action_root = Path("dataset\\bad_leg")
checkpoint_root.mkdir(exist_ok=True)
action_root.mkdir(exist_ok=True)


params = dict()
params["model_folder"] = "..\openpose\models"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# socket set up
HOST='192.168.0.132'
PORT=5555

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))

#cap = cv2.VideoCapture(0)
keypoints_list = []
while True:
    pre_time = time.time()
    while len(data) < payload_size:
        data += conn.recv(64)
        if not data:
            cv2.destroyAllWindows()
            conn,addr=s.accept()
            continue
    # receive image row data form client socket
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(64)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    
    # unpack image using pickle 
    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    


    #ret, frame = cap.read()

    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    #show_frame = np.concatenate((frame, datum.cvOutputData), axis=1)
    # if datum.poseKeypoints != None:

    #print(datum.poseKeypoints[0])
    print("----------------------------------")
    if record_mode == True:
        keypoints_list.append(datum.poseKeypoints[0])
        if save_count%10 == 0:
            np.save(r"dataset\\bad_leg\\data", np.array(keypoints_list))
            print("dataset\\bad_leg\\data")
    else:

    cv2.imshow('img', datum.cvOutputData)
    #print(f'\r{1/ (time.time() - pre_time)}')
    save_count += 1
    # exit()
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
