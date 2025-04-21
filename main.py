import torch
import numpy as np
from args_parse import Args
from utils import make_model, load_model, generateParameters, SeqSetting
from dataloader import read_data, read_test_data
from train import train
from test import test_model, interval_plot
import random
import time
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from module import enrich_para
##
import pdb


if __name__ == '__main__':
    ## fix the random seeds
    
    print('starts')
    seed_num = 0
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)
    #args = Args()
    kwargs = Args().update_args()
    print('Using device ', kwargs.device)

    if kwargs.dataset == 'cylinder':
        numT = 401
    elif kwargs.dataset == 'vas2d':
        numT = 250
    elif kwargs.dataset == 'sine':
        numT = 1000
    elif kwargs.dataset == 'stBurgers':
        numT = 401
    elif kwargs.dataset == 'stinitial_stBurgers':
        numT = 401
    elif kwargs.dataset == 'testsmallstBurgers':
        numT = 401*2
    elif kwargs.dataset == 'gluon_test':
        numT = 401
    elif kwargs.dataset == 'mvw':
        numT = 40
    elif kwargs.dataset == 'les2d':
        numT = 240
    elif kwargs.dataset == 'les2d30':
        numT = 240
        #numT = 120
    else:
        raise TypeError('Only cylinder and sine and stBurgers are supported')


    #pdb.set_trace()
    # input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, prediction_step, mu_size,device="cuda:0"
    #config =  SeqSetting()
    #model = make_model(kwargs.input_size, kwargs.num_layers, kwargs.hidden_size, kwargs.num_cells, kwargs.dropout_rate, kwargs.n_blocks, kwargs.n_hidden, kwargs.mu_size, adj_var = kwargs.adj_var, device = kwargs.device, conditioning_length= kwargs.conditioning_length,
                       #config = config)  #TODO: FINISH IT
    model = make_model(kwargs.input_size, kwargs.num_layers, kwargs.hidden_size, kwargs.num_cells, kwargs.dropout_rate, kwargs.n_blocks, kwargs.n_hidden, kwargs.mu_size, adj_var = kwargs.adj_var, device = kwargs.device, conditioning_length= kwargs.conditioning_length,
                       num_heads = kwargs.num_heads, dim_feedforward_scale = kwargs.dim_feedforward_scale,
                        num_encoder_layers = kwargs.num_encoder_layers, num_decoder_layers = kwargs.num_decoder_layers, mask_length = int(numT-1-kwargs.startT), d_model = kwargs.d_model)
    if kwargs.transfer_flag == 1:
        print('\n')
        print('load_transfer epoch ', kwargs.transfer_epoch)
        print('\n')
        time.sleep(2)
        ## the first 15 characters are transferXXXXXX_, for trasnfer learning, we set 10 times lower lr and 
        ## batch 1 for fine tuning
        savedir = 'saved_model/para_'+str(kwargs.modelref)[15:]+'/'
        model.load_state_dict(torch.load(savedir+"model_"+str(kwargs.transfer_epoch)))

    if kwargs.train == 1:
        print('training dataset')
        filePath = "data/"+str(kwargs.dataset)+"_noise"+str(kwargs.noise)+"_dataSave.txt"
    #pdb.set_trace()
    ###
    
    ###
    elif kwargs.train == 0:
        print('test dataset')
        filePath = "data/test_"+str(kwargs.dataset)+"_noise"+str(kwargs.noise)+"_dataSave.txt"
    if kwargs.dataset != 'gluon_test':
        
        dataAll = read_data(path=filePath, device = kwargs.device)
        
        #pdb.set_trace()
        ##
        '''
        ### only use the first case
        dataDouble = dataAll[:401*2,:]
        np.savez('data/DoubleData.npz', dataDouble= dataDouble.cpu().detach().numpy())
        
        pdb.set_trace()
        '''
        '''
        testData = np.load('data/DoubleData.npz')
        dataAll = torch.from_numpy(testData['dataDouble']).to(kwargs.device).float()
        #pdb.set_trace()
        ## only learn 1 spatial points
        dataAll = dataAll[:,0:10]
        #pdb.set_trace()
        '''
        ###
    else:
        datalist = read_test_data(device = kwargs.device)
    #pdb.set_trace()
    print('shape of all the data is', dataAll.shape)
    time.sleep(2)
    ## paradata

    
    #pdb.set_trace()
    if kwargs.dataset != 'gluon_test':
        data = []
        #pdb.set_trace()
        for i in range(kwargs.mu_size):     
        #for i in range(int(dataAll.shape[0]/numT)):
            tmp = dataAll[i*numT:(i+1)*numT].unsqueeze(0)
            data.append(tmp)


        data = torch.cat(data, dim = 0)
        ## let the data start from some point
        data = data[:,kwargs.startT:,:]
        #pdb.set_trace()
        print('size of data before is', data.shape)
        # ## pick every 2 cases
        # shape0 = 2*kwargs.mu_size
        # data = data[range(1,shape0,2),:]
        
        print('size of data is', data.shape)
        #pdb.set_trace()
        ## normalize data
        mean_d = data.mean(dim = (1,2), keepdim = True)
        std_d = data.std(dim = (1,2), keepdim = True)
        #pdb.set_trace()
        data = (data-mean_d)/std_d

        
        
        ##
        #pdb.set_trace()
        print('data is normalized, shape is', data.shape)
        print('\n')
        time.sleep(1.5)


        ##
    else:
        dataset = IterableDataset(datalist)
        data = DataLoader(datalist, batch_size = kwargs.batchsize)
        #temp1 = next(iter(data))
        for k,v in (next(iter(data))).items():
            print(k+' shape is', v.shape)
    #pdb.set_trace()
    #data=data[0:400].unsqueeze(0)

    ## select parameters

    #pdb.set_trace()
    print('----------------data loaded for case "'+str(kwargs.dataset)+'"-------------')
    print('------------- lr rate is----------------'+str(kwargs.lr))
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
	                                            step_size = 1.0,
												gamma =1.0)
    
    
    para_mu = generateParameters(kwargs.dataset, kwargs.mu_size, kwargs.device, token = kwargs.token, train_flag = kwargs.train)
    if kwargs.token == True:
        print('with token')
        paraOI = enrich_para(para_mu)
    elif kwargs.token == False:
        print('no token')
        paraOI = None
    
    #datatuple = {'paraOI':paraOI, 'data':data}
    ##
    ## paraOI [b,6], data[b,T,c]
    ##
    if kwargs.token == True:
        datatuple = [(paraOI[i], data[i]) for i in range(len(paraOI))]
    elif kwargs.token == False:
        datatuple = data
    
    #pdb.set_trace()
    #dataloader = DataLoader(datatuple, batch_size=kwargs.batchsize,shuffle=True,num_workers = 0,drop_last = True )
    print('shuffle flag is', kwargs.shuffle)
    dataloader = DataLoader(datatuple, batch_size=kwargs.batchsize,shuffle=kwargs.shuffle,num_workers = 0,drop_last = True )
    #pdb.set_trace()
    if kwargs.train == 1:
    
        #para_mu = generateParameters(kwargs.dataset, kwargs.mu_size, kwargs.device, token = kwargs.token)
        #pdb.set_trace()
        ##
        print('----------------training start-------------')
        start = time.time()
        train(dataloader, model, kwargs.epoch, optimizer, modelref = kwargs.modelref, scheduler = scheduler)
        print('----------------training end-------------')
        print('elapse time is ',time.time()-start)
    elif kwargs.train== 0:
        print('----------------test start-------------')
        #pdb.set_trace()
        model = load_model(model, 'saved_model/para_'+str(kwargs.modelref)+'/model_'+str(kwargs.test_epoch))
        #model = load_model(model, 'saved_model/'+str(kwargs.modelref)+'/model_'+str(kwargs.test_epoch))
        #x=data[:,0].unsqueeze(1)
        
        x = data[:, :kwargs.history_length]
        #pdb.set_trace()
        endidx=  min(data.shape[1], kwargs.history_length+kwargs.prediction_length)
        ground=data[:,kwargs.history_length:endidx]
        #try:
            #interval_plot(kwargs.modelref, x, ground, kwargs.test_epoch, test_samples = kwargs.test_samples, caseName = kwargs.dataset, prediction_length = kwargs.prediction_length, history_length = kwargs.history_length,sample_flag = kwargs.sample_flag)
        #except:
        test_model(model, x, kwargs.history_length,kwargs.prediction_length, ground, caseName = kwargs.dataset, epoch = kwargs.test_epoch, test_samples = kwargs.test_samples, modelref = kwargs.modelref, paraOI = paraOI, sample_flag = kwargs.sample_flag)
        interval_plot(kwargs.modelref, x, ground, kwargs.test_epoch, test_samples = kwargs.test_samples, caseName = kwargs.dataset, prediction_length = kwargs.prediction_length, history_length = kwargs.history_length,sample_flag = kwargs.sample_flag)
        print('----------------test finish-------------')
        #test()
    





