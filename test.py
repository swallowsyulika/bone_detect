import numpy as np

train_label_filepath = r'dataset\\bad_leg\\data.npy'

data = np.load(train_label_filepath, allow_pickle=True)


for i in range(len(data)):
    print(data)
    print("----------------------------------------------------")