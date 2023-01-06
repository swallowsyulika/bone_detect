from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, dir_list, transform=None):
        self.dir_list = dir_list
        self.transform = transform

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
        max_list = np.amax(datas, axis=2)
        max = np.amax(max_list, axis=0)
        max_x = max[0]
        max_y = max[1]
        for i in range(len(datas)):
            datas[i][:][0] = datas[i][:][0]/max_x
            datas[i][:][1] = datas[i][:][1]/max_y      
        print(datas.shape)
        
        return datas, labels
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):

        data = self.datas[idx]
        label = self.labels[idx]
        if self.transform:
            data = self.transform(data)   
        return data, label


if __name__ == "__main__":
    ds = MyDataset(['normal', 'bad_leg'])
    print(ds[5])