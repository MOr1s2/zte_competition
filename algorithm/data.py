import torch
import pandas as pd
from torch.utils.data import Dataset


class myDataset(Dataset):
    def __init__(self, path, mode='train'):
        self.mode = mode
        self.dataset = pd.read_csv(path).drop(columns=['sample_id'])
        self.preProcess()

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        if self.mode == 'train':
            x = torch.tensor(self.dataset.values[index,:-1])
            y = torch.tensor(self.dataset.values[index,-1])
            return x, y
        else:
            x = torch.tensor(self.dataset.values)
            return x


    def preProcess(self):
        if self.mode == 'train':
            df = self.dataset.drop(columns=['label'])

            df = (df - df.min()) / ( df.max() - df.min())

            mean = df.mean(axis=0)
            df.fillna(mean, inplace=True)

            df.drop(df.columns[[2, 4, 17, 23, 30, 36, 38, 41, 48, 58, 64, 70, 79, 86]], axis=1, inplace=True)
            
            df = pd.concat([df, self.dataset['label']], axis=1)

            del_nums = [2200, 0, 0, 0, 0, 0]     
            add_nums = [0, 400, 0, 500, 600, 700]
            del_rows = []
            add_rows = []
            for index, row in df.iterrows():
                for i in range(6):
                    if row['label'] == i and del_nums[i] != 0:
                        del_rows.append(index)
                        del_nums[i] -= 1
                    if row['label'] == i and add_nums[i] != 0:
                        add_rows.append(index)
                        add_nums[i] -= 1
                    if not any(del_nums) and not any(add_nums):
                        break
            self.dataset = pd.concat([df, df.iloc[add_rows]])
            self.dataset.drop(del_rows, inplace=True)
            print(f"Train set:\n{self.dataset['label'].value_counts()}")

        else:
            df = self.dataset

            df = (df - df.min()) / ( df.max() - df.min())

            mean = df.mean(axis=0)
            df.fillna(mean, inplace=True)

            df.drop(df.columns[[2, 4, 17, 23, 30, 36, 38, 41, 48, 58, 64, 70, 79, 86]], axis=1, inplace=True)

            self.dataset = df