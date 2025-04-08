import os

import torch
# import numpy as np
import torch.nn.functional as fun
from torch.utils.data import Dataset  # For constructing datasets, supporting indexing and total length
# from torch.utils.data import DataLoader

from utils.readGrd import readGrdbynp



class Basement(Dataset):

    def __init__(self, root: str):  # root specifies the root directory, resize customizes data size
        super(Basement, self).__init__()
        self.root = root
        assert os.path.join(self.root), f"path '{self.root}' does not exists."  # Check if file path exists

        dg_names = [i for i in os.listdir(os.path.join(self.root)) if i.endswith(".grd")]
        self.dg_list = [os.path.join(self.root, i) for i in dg_names]  # Get all grd files under the dg folder

        # check files
        for i in self.dg_list:  # Note that in the for i in self.dg_list or dg_names, i is a string type not an integer
            if os.path.exists(i) is False:
                raise FileExistsError(f"file {i} doesn't exissts.")

    def __getitem__(self, index):  # Get the sample and label corresponding to the current index in the file list
        dg = readGrdbynp(self.dg_list[index])
        dg = torch.from_numpy(dg).unsqueeze(0).unsqueeze(0)
        # print(mod_label.shape,dg.shape)
        # Resample data to reconstruct to network target input size 64*64, 64*64, 2D data interpolate function input needs to be [b,c,h,w]
        dg_sample = fun.interpolate(dg, size=[64, 64], mode='bilinear', align_corners=True)
        # Remove batch dimension
        dg_sample = dg_sample.squeeze(0)

        return dg_sample.float()

    def __len__(self):
        return len(self.dg_list)


# # # Test if dataset reading is successful
if __name__ == '__main__':
    myroot = r"D:\Project\DCGAN\data\real"
    mydata = Basement(myroot)
    print(mydata.__len__())
    print(mydata.dg_list[0])
    print(mydata[0].shape, torch.max(mydata[0]))
