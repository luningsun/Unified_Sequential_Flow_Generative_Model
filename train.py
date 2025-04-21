import torch
#from sequential_flow import PhysicsFlow,PhysicsFlow_residual
import numpy as np
from utils import save_model
from pathlib import Path
#from utils import read_data, make_model
import pdb
import time



def train(dataloader,model,steps,optimizer, modelref = 3, scheduler= None):
    ##
    savedir = 'saved_model/para_'+str(modelref)+'/'
    Path(savedir).mkdir(exist_ok = True, parents = True)
    ##
    tic = time.time()
    #nT = 399
    
    for i in range(steps+1):
        #tic=time.time()
        #print('Start epoch = ',i)
        #print('nT = ',nT)
        for batch_idx, bulk in enumerate(dataloader):
            #pdb.set_trace()
            #print(batch_idx)
            if type(bulk) == list:
                paraOI, data = bulk[0], bulk[1]
            else:
                paraOI, data = None, bulk
            #print('batch_idx is', batch_idx)
            #pdb.set_trace()
            optimizer.zero_grad()
            #loss, Er = model(data, para_mu, nT = nT)
            
            loss = model(data, paraOI)
            loss = torch.mean(loss)
            ## add scheduler not sure if it is good for generative models

            # print('Last LR is '+str(scheduler.get_last_lr()))
            # print('TrainData Hid Error max = ',max(Er))
            # if scheduler is not None and max(Er) < 0.05:
            #     scheduler.step()
            #     save_model(model,savedir+"model_"+str(i))
            #     nT = nT+1
            #     if nT == data.shape[1]+1:
            #         exit()
            # ##
            loss.backward()
            optimizer.step()
        #print('Elapsed time = ',time.time()-tic)	
        if i%100 == 0:
            print('elapse_time is', time.time() - tic)
            print("steps {}: mean loss is {}".format(i, loss))
            tic = time.time()
        if i%1000 == 0:
            save_model(model,savedir+"model_"+str(i))


# def main():
#     print('hello world')
#     data = read_data(path="data/cylinder_noise0.0_dataSave.txt")
#     data=data[1:400].unsqueeze(0)
#     model=make_model(input_size=1024, num_layers=3, hidden_size=1024, num_cells=1024, dropout_rate=0.0,n_blocks=4, n_hidden=3, prediction_step=99, mu_size=6)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     train(data,model,20000, optimizer)

# main()