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
from model import Net
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from threading import Thread
from myDataset import MyDataset

record_mode = False
save_count = 0

if record_mode == False:
    weights_root = Path("weights")
    weight = 'E050.pth'
    PATH = os.path.join(weights_root, weight)
    

checkpoint_root = Path("dataset")
action_root = Path("dataset\\bad_leg")
checkpoint_root.mkdir(exist_ok=True)
action_root.mkdir(exist_ok=True)

torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))


params = dict()
params["model_folder"] = "..\openpose\models"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# socket set up
HOST='192.168.0.159'
PORT=5555
HOST2='192.168.0.142'
PORT2=5556

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s2=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

print('Socket created')

s.bind((HOST,PORT))
s2.connect((HOST2, PORT2))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))

transforms_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])


# model set up
if record_mode==False:
    net  = Net()
    net.to(device)
    net.load_state_dict(torch.load(PATH))
    
def send_result(result_string,s2):
    cmd = result_string
    s2.send(cmd.encode("utf-8"))
    
def dataPreprocess(data):
    # print(data.shape)
    

    max = np.amax(data, axis=0)
    # print(max)
    np.transpose(data)[0] = np.transpose(data)[0]/max[0]
    np.transpose(data)[1] = np.transpose(data)[1]/max[1]
    # print('+++++++++++++++++++++++++++')
    # print(data)
    data = transforms_(data).to(device)
    data = torch.permute(data,(0, 2, 1))
    return data

##
# cap = cv2.VideoCapture(0)
keypoints_list = []

def dataset_path(normal ,file):
    if normal:
        path = f"dataset\\normal\\{file}"
    else:
        path = f"dataset\\bad_leg\\{file}"
    return path

def main():
    while True:
        global data, conn, save_count
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


        #print(frame.shape) ##(480, 640, 3)
        # ret, frame = cap.read()

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        #show_frame = np.concatenate((frame, datum.cvOutputData), axis=1)
        # if datum.poseKeypoints != None:

        #print(datum.poseKeypoints[0])
        print("----------------------------------")
        print(save_count)

        if datum.poseKeypoints is None:
            continue
        if record_mode == True:
            keypoints_list.append(datum.poseKeypoints[0])
            if save_count%10 == 0:
                path = dataset_path(False, 'data2')
                np.save(path, np.array(keypoints_list))
                print(path)
        else:
            ## recognition action

            input = datum.poseKeypoints[0]
            #print("keypoint 4", input[4][1])
            #input = dataPreprocess(input)
            input = MyDataset(single_img=input, transform=transforms_)[0]

            #torch.from_numpy(data).to(device)
            #print("transform : ",data.shape)
            output = net(input)
            #print(output)
            output = torch.argmax(output, dim=1)
            mythread = Thread(target = send_result, args=(str(output),s2))
            if output == 1:
                print('Good')
            elif output == 0:
                print('U R so Bad')


        cv2.imshow('img', datum.cvOutputData)
        #print(f'\r{1/ (time.time() - pre_time)}')
        save_count += 1
        # exit()
        if cv2.waitKey(1) & 0xFF == 27:
            break

main()

#cap.release()
cv2.destroyAllWindows()
