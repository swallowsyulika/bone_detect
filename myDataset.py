from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, dir_list=None, single_img=None, transform=None):
        self.dir_list = dir_list
        self.single_img = np.expand_dims(single_img.T, axis=0) if single_img is not None else None

        self.transform = transform
        self.center = (320, 240)
        self.threshold = 0.2

        #assert dir_list != single_img, "need dir_list or single_img"
        self.eval = False if dir_list is not None else True

        if self.eval:
            self.datas = self.dataPreprocess(single_img)
        else:
            self.datas, self.labels = self.load_data(dir_list)


    def load_data(self, class_list):
        datas = []
        labels = []
        for name in class_list:
            print(name)
            for i in range(3):
                
                data = np.load(f'dataset\\{name}\\data{i}.npy', allow_pickle=True)
                
                if name == 'normal':
                    label = np.ones(len(data))
                else:
                    label = np.zeros(len(data))
                if len(datas) == 0:
                    datas = data
                    labels = label
                else:

                    datas = np.concatenate([datas, data],axis=0)
                    labels = np.concatenate([labels, label])

        datas = np.transpose(datas, (0, 2, 1)) 
        print(datas[0].shape)
        datas = self.dataPreprocess(datas)
        return datas, labels

    def dataPreprocess(self, datas):

        for i in range(len(datas)):
            datas[i][:][0] = datas[i][:][0] - self.center[0]
            datas[i][:][1] = datas[i][:][1] - self.center[1]

        # max_list = np.amax(datas, axis=2)
        # max = np.amax(max_list, axis=0)
        # max_x = max[0]
        # max_y = max[1]
        for i in range(len(datas)):
            single_data = np.expand_dims(datas[i], axis=0)
            max_list = np.amax(single_data, axis=2)
            max = np.amax(max_list, axis=0)
            max_x = max[0]
            max_y = max[1]

            datas[i][:][0] = datas[i][:][0]/max_x
            datas[i][:][1] = datas[i][:][1]/max_y
            for idx, ele in enumerate(datas[i][:][2]):
                if ele < self.threshold:
                    datas[i][:][0][idx] = 0.0
                    datas[i][:][1][idx] = 0.0

        print(datas.shape)
        
        return datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if self.eval:
            data = self.datas[idx]
            if self.transform:
                data = self.transform(data)  
            return data
        else: 
            data = self.datas[idx]
            label = self.labels[idx]
            if self.transform:
                data = self.transform(data)   
            return data, label


if __name__ == "__main__":
    ds = MyDataset(dir_list=['normal', 'bad_leg'])
    print(ds[7][0])
    data = np.load("dataset/normal/data0.npy")
    img = np.expand_dims(data[7].T, axis=0)
    ins = MyDataset(single_img=img)[0]
    print(ins)