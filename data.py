import torch
import numpy as np



def read_data(path='transformerTrainingData.txt', device='cuda:0'):
    data = torch.from_numpy(np.loadtxt(path)).to(device).float()   #(20050)
    return data

def read_data(data_name, device='cuda:0'):
    pass