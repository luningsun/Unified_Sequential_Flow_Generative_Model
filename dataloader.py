import torch
import numpy as np
import pdb
from torch.utils.data import Dataset, IterableDataset, DataLoader
from itertools import chain, cycle, islice

def read_data(path='transformerTrainingData.txt', device='cuda:0'):
    data = torch.from_numpy(np.loadtxt(path)).to(device).float()   #(20050)
    #pdb.set_trace()
    if len(data.shape) == 1:
        data = data.unsqueeze(1)
    #pdb.set_trace()
    return data

def read_test_data(device = 'cuda:0'):
    print()
    savedir = '/home/luningsun/storage/gnn_flow/pytorch-ts-master/examples/torchData/'
    
    nameList = ['target_dimension_indicator', 'past_time_feat', 'past_target_cdf', 'past_observed_values', 'past_is_pad', 'future_time_feat','future_target_cdf', 'future_observed_values']
    
    data0 = np.load(savedir+nameList[0]+'.npz')
    data1 = np.load(savedir+nameList[1]+'.npz')
    data2 = np.load(savedir+nameList[2]+'.npz')
    data3 = np.load(savedir+nameList[3]+'.npz')
    data4 = np.load(savedir+nameList[4]+'.npz')
    data5 = np.load(savedir+nameList[5]+'.npz')
    data6 = np.load(savedir+nameList[6]+'.npz')
    data7 = np.load(savedir+nameList[7]+'.npz')
    target_dimension_indicator = torch.Tensor(data0['v']).to(device)
    past_time_feat = torch.Tensor(data1['v']).to(device)
    past_target_cdf = torch.Tensor(data2['v']).to(device)
    past_observed_values = torch.Tensor(data3['v']).to(device)
    past_is_pad = torch.Tensor(data4['v']).to(device)
    future_time_feat = torch.Tensor(data5['v']).to(device)
    future_target_cdf = torch.Tensor(data6['v']).to(device)
    future_observed_values = torch.Tensor(data7['v']).to(device)
    dataList = [target_dimension_indicator, past_time_feat, past_target_cdf, past_observed_values, past_is_pad, future_time_feat, future_target_cdf, future_observed_values] 
    #pdb.set_trace()
    return dataList
#def read_data(data_name, device='cuda:0'):
    #pass

class MyIterableDataset(IterableDataset):
    def __init__(self, data_list, batch_size):
        self.data_list = data_list
        self.batch_size = batch_size

    def process_data(self, data):
        for x in data:
            yield x

    def get_stream(self, data_list):
        return chain.from_iterable(map(self.process_data, cycle(data_list)))

    def get_streams(self):
        return zip(*[])

    def __iter__(self):
        return self.get_stream(self.data_list)
    
if __name__ == '__main__':
    device = 'cuda:0'
    ## not important at this time, change model first
    #data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    data = read_test_data(device = device)
    print(data[0].shape)
    '''
    data = [
        [12, 13, 14, 15, 16, 17],
        [27, 28, 29],
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        [40, 41, 42, 43],
    ]
    '''
    iterable_dataset = MyIterableDataset(data)
    loader = DataLoader(iterable_dataset, batch_size = 4)

    for batch in islice(loader,3):
        print(batch.shape)

