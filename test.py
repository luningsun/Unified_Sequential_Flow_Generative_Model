import torch
from sequential_flow import PhysicsFlow, PhysicsFlow_residual
import numpy as np
import pdb
from pathlib import Path
from matplotlib import pyplot as plt
import time
from dataloader import read_data
# def read_data(path='transformerTrainingData.txt', device='cuda:0'):
#     data = torch.from_numpy(np.loadtxt(path)).to(device).float()   #(20050)
#     return data

# def make_model(input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, prediction_step, mu_size,device="cuda:0"):
#     model = PhysicsFlow(input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, prediction_step, mu_size)
#     # model = PhysicsFlow_residual(input_size, num_layers, hidden_size, num_cells, dropout_rate, n_blocks, n_hidden,
#     #                     prediction_step, mu_size)
#     return model.to(device)



# def load_model(model,path):
#     model.load_state_dict(torch.load(path))
#     model.eval()
#     return model
def ensemble_plot(modelref, Alldata,x, ground_trueth,sample_flag,prediction_length,epoch):
    tic = time.time()
    for ntraj in range(x.shape[0]):
        
        print('elapse time is', time.time()-tic)
        tic = time.time()
        
        for nsamples in range(Alldata.shape[0]):
            testPtplot = 'testplot/'+str(modelref)+'/sampleFlag'+str(sample_flag)+'/ensemble/traj'+str(ntraj)+'/pred'+str(prediction_length)+'/hist'+str(x.shape[1])+'/Epoch'+str(epoch)+'/Ensemble'+str(nsamples)+'/'
            Path(testPtplot).mkdir(exist_ok = True, parents = True)
            print('ensemble plot for case ', str(ntraj), 'ensemble ', str(nsamples))
            for testPt in range(x.shape[2]):
                if testPt % 200 ==0:
                    print('plot for pt ', str(testPt))
                #pdb.set_trace()
                mean_predict = Alldata[nsamples, ntraj, :prediction_length, testPt]

                tplot = np.array(range(x.shape[1]+max(ground_trueth.shape[1], prediction_length)))
                #pdb.set_trace()
                plt.figure()
                plt.plot(tplot[:x.shape[1]], (x[ntraj,:,testPt]).cpu().detach().numpy(), 'r-', label = 'History')
                endidx = min(x.shape[1]+ground_trueth.shape[1], x.shape[1]+prediction_length)
                #pdb.set_trace()
                plt.plot(tplot[x.shape[1]:endidx], ground_trueth[ntraj,:,testPt].cpu().detach().numpy(), 'r-.', label = 'Future')
                plt.plot(tplot[x.shape[1]:], mean_predict, 'b--',label = 'Predict Mean')
                
                
                
            
                criterion    = torch.nn.MSELoss()
                #pdb.set_trace()
                eU = torch.sqrt(criterion(torch.from_numpy(mean_predict[2:]).to('cuda:0'), ground_trueth[ntraj,:-2,testPt])/\
                                            criterion(ground_trueth[ntraj,:-2,testPt]*0, ground_trueth[ntraj,:-2,testPt]))
            
                plt.title('relative error '+str(eU.item()))
                
                plt.savefig(testPtplot+'Case'+str(ntraj)+'_Epoch'+str(epoch)+'_testPoint'+str(testPt)+'Ensemble'+str(nsamples)+'.png', bbox_inches = 'tight')
                
                #pdb.set_trace()
                plt.close('all')


def interval_plot(modelref, x, ground_trueth, epoch, test_samples = 1, caseName = None, prediction_length = 100, history_length = 300, sample_flag = True):
    testdir = 'testres/'+str(modelref)+'/sampleFlag'+str(sample_flag)+'/'
    ## [nsample, ntraj, ntime, nspatial]
    #pdb.set_trace()
    datanpz = np.load(testdir+caseName+'_epoch'+str(epoch)+'_samples'+str(test_samples-1)+'_pred'+str(prediction_length)+'_hist'+str(history_length)+'_Alldata.txt.npz')
    Alldata = datanpz['Alldata']
    if Alldata.shape[0]==1:
        mat_std = Alldata.std(0,keepdims = True)
        mat_mean = Alldata.mean(0, keepdims = True)
    else:
        mat_std = Alldata.std(0)
        mat_mean = Alldata.mean(0)
    #pdb.set_trace()
    
    
    #ensemble_plot(modelref, Alldata,x,ground_trueth,sample_flag,prediction_length,epoch)
    tic = time.time()
    for ntraj in range(x.shape[0]):
        print('interval plot for case ', str(ntraj))
        print('elapse time is', time.time()-tic)
        tic = time.time()
        testPtplot = 'testplot/'+str(modelref)+'/sampleFlag'+str(sample_flag)+'/interval/traj'+str(ntraj)+'/pred'+str(prediction_length)+'/hist'+str(x.shape[1])+'/Epoch'+str(epoch)+'/'
        Path(testPtplot).mkdir(exist_ok = True, parents = True)
        for testPt in range(x.shape[2]):
            if testPt % 100 ==0:
                print('plot for pt ', str(testPt))
                #pdb.set_trace()
                std_predict = 1.96*mat_std[ntraj, :prediction_length, testPt]
                mean_predict = mat_mean[ntraj, :prediction_length, testPt]
                tplot = np.array(range(x.shape[1]+max(ground_trueth.shape[1], prediction_length)))
                #pdb.set_trace()
                plt.figure()
                plt.plot(tplot[:x.shape[1]], (x[ntraj,:,testPt]).cpu().detach().numpy(), 'r-', label = 'History')
                endidx = min(x.shape[1]+ground_trueth.shape[1], x.shape[1]+prediction_length)
                #pdb.set_trace()
                plt.plot(tplot[x.shape[1]:endidx], ground_trueth[ntraj,:,testPt].cpu().detach().numpy(), 'r-.', label = 'Future')
                plt.plot(tplot[x.shape[1]:], mean_predict, 'b--',label = 'Predict Mean')
                
                
                plt.fill_between(tplot[x.shape[1]:], mean_predict-std_predict, mean_predict+std_predict, color='g', alpha=.25)
            
                criterion    = torch.nn.MSELoss()
                #pdb.set_trace()
                eU = torch.sqrt(criterion(torch.from_numpy(mean_predict[2:]).to('cuda:0'), ground_trueth[ntraj,:-2,testPt])/\
                                            criterion(ground_trueth[ntraj,:-2,testPt]*0, ground_trueth[ntraj,:-2,testPt]))
            
                plt.title('relative error '+str(eU.item()))
                #pdb.set_trace()
                # plt.fill_between(tplot[x.shape[1]:x.shape[1]+100], (mean_predict-std_predict)[:100], (mean_predict+std_predict)[:100], color='y', alpha=.8)
                # if prediction_length>100:
                #     plt.fill_between(tplot[x.shape[1]+100:], (mean_predict-std_predict)[100:], (mean_predict+std_predict)[100:], color='g', alpha=.25)
                plt.savefig(testPtplot+'Case'+str(ntraj)+'_Epoch'+str(epoch)+'_testPoint'+str(testPt)+'.png', bbox_inches = 'tight')
                plt.figure()
                #plt.plot(mean_predict[2:]-ground_trueth[ntraj,:-2,testPt].cpu().detach().numpy())
                plt.plot(mean_predict-ground_trueth[ntraj,:,testPt].cpu().detach().numpy())
                plt.title('pointpise diff predict-trueth')
                plt.savefig(testPtplot+'Case'+str(ntraj)+'_Epoch'+str(epoch)+'_testPoint'+str(testPt)+'difference.png', bbox_inches = 'tight')
                plt.figure()
                #plt.plot(ground_trueth[ntraj,-100:-2,testPt].cpu().detach().numpy(), 'r-x', label = 'Future')
                plt.plot(ground_trueth[ntraj,:,testPt].cpu().detach().numpy(), 'r-x', label = 'Future')
                #pdb.set_trace()
                #plt.plot(mean_predict[-98:], 'b-x',label = 'Predict Mean')
                plt.plot(mean_predict, 'b-x',label = 'Predict Mean')
                plt.savefig(testPtplot+'Case'+str(ntraj)+'_Epoch'+str(epoch)+'_testPoint'+str(testPt)+'last100compare.png', bbox_inches = 'tight')
                #pdb.set_trace()
                plt.close('all')
            

# def test_plot(modelref, x,ground_trueth, samples, numsample, epoch, prediction_length = 100):
#     #pdb.set_trace()
#     for ntraj in range(x.shape[0]):
#         testPtplot = 'testplot/'+str(modelref)+'/samples'+str(numsample)+'/traj/'+str(ntraj)+'/pred'+str(prediction_length)+'/ptPlot/'
#         Path(testPtplot).mkdir(parents = True, exist_ok = True)
#         #pdb.set_trace()
#         for testPt in range(x.shape[2]):
#             print('plotting for point ', str(testPt))
#             plt.figure()
#             tplot = np.array(range(x.shape[1]+ground_trueth.shape[1]))
#             #plt.plot(u_trunc[-npredict:,testPt], label = 'true')
#             #pdb.set_trace()
#             #pdb.set_trace()
#             plt.plot(tplot[:x.shape[1]], (x[ntraj,:,testPt]).cpu().detach().numpy(), 'r-', label = 'History')
            
#             plt.plot(tplot[x.shape[1]:x.shape[1]+100], ground_trueth[ntraj,:100,testPt].cpu().detach().numpy(), 'r-.', label = 'Seen Valid')
#             plt.plot(tplot[x.shape[1]:x.shape[1]+100], samples[ntraj,:100,testPt].cpu().detach().numpy(), 'b--',label = 'Predict Valid')
#             if prediction_length > 100:
#                 endidx = min(x.shape[1]+ground_trueth.shape[1], x.shape[1]+prediction_length)
#                 plt.plot(tplot[x.shape[1]+100:endidx], ground_trueth[ntraj,100:,testPt].cpu().detach().numpy(),'rx', label = 'Future Extra')
#                 plt.plot(tplot[x.shape[1]+100:], samples[ntraj,100:,testPt].cpu().detach().numpy(), 'bx',label = 'Predict Extra')
#             plt.legend()
#             plt.savefig(testPtplot+'Case'+str(ntraj)+'_Epoch'+str(epoch)+'_testPoint'+str(testPt)+'.png', bbox_inches = 'tight')
#             #pdb.set_trace()
#             plt.close('all')



def test_model(model, x, history_length, prediction_length, ground_trueth=None, caseName = None, epoch = None, test_samples = 1, modelref = None, paraOI = None, sample_flag = True):
    #testdir = 'testres/'+str(modelref)+'/'
    testdir = 'testres/'+str(modelref)+'/sampleFlag'+str(sample_flag)+'/'
    #pdb.set_trace()
    Path(testdir).mkdir(exist_ok = True, parents = True)
    Alldata = []
    AllrmseU = []
    maxrmseU = []
    meanrmseU = []
    sample_list = []
    with torch.no_grad():
        #pdb.set_trace()
        for i in range(test_samples):
            
            print('before sample')
            #pdb.set_trace()
            #samples, temp_list = model.sample(x, prediction_length)
            samples = model.sample(x, prediction_length, paraOI = paraOI, sample_flag = sample_flag)
            #pdb.set_trace()
            ground_trueth=ground_trueth[:,:prediction_length]
            #pdb.set_trace()
            
            endidx = min(ground_trueth.shape[1], samples.shape[1])
            mse = torch.nn.MSELoss(reduction='mean')(samples[:,:endidx],ground_trueth)
            print("finish test sample "+str(i+1)+"! mse is " +str(mse.item()))
            np.savetxt(testdir+'Epoch'+str(epoch)+'_mse.txt', [mse.cpu().detach().numpy()])
            #pdb.set_trace()
            ## test the rmse error
            EU = []
            criterion    = torch.nn.MSELoss()

            for nC in range(samples.shape[0]):
                eU = torch.sqrt(criterion(samples[nC], ground_trueth[nC])/\
                                        criterion(ground_trueth[nC]*0, ground_trueth[nC]))
                EU.append(eU.item())
            #pdb.set_trace()
            maxEU = max(EU)
            meanEU = sum(EU)/len(EU)
                

            print('EU max = ', max(EU))
            print('Ux Error (mean) = ',sum(EU)/len(EU))

            


            AllrmseU.append(EU)
            #pdb.set_trace()
            #maxrmseU.append(eU)
            
            #pdb.set_trace()
            ##

            #test_plot(modelref, x, ground_trueth, samples, i, epoch, prediction_length = prediction_length)
            
            #print("finish test sample {}! mse is {}",format(i+1, mse))
            samples= samples.cpu().squeeze(0).detach().numpy()
            Alldata.append(samples)
            ## convert to 2d
            if len(samples.shape)>=3:
                fsamples = np.array([], dtype=np.int64).reshape(0,samples.shape[-1])
                for p in range(samples.shape[0]):
                    fsamples = np.vstack([fsamples,samples[p]])
                #fsamples = np.array(fsamples)
            else:
                fsamples = samples    
            ##
            #pdb.set_trace()
            
            np.savetxt(testdir+caseName+'_epoch'+str(epoch)+'_samples'+str(i)+'_latent_pred_'+str(prediction_length)+'_start_'+str(history_length)+'.txt',fsamples)
            #np.savetxt('testres/'+str(modelref)+'/'+caseName+'_epoch'+str(epoch)+'_samples'+str(i)+'_latent_pred_100_start_300.txt',samples)
            #np.savetxt('testres/'+caseName+'_epoch'+str(epoch)+'_model2_mean_latent_100_start_200.txt',samples)
            ## calculate the metrics
            maxrmseU.append(maxEU)
            meanrmseU.append(meanEU)
            #sample_list.append(temp_list)
        
        
        #pdb.set_trace()
        ## sample the attention
        #sample_list = torch.cat(sample_list[0])
        #pdb.set_trace()

        ##
        np.savetxt(testdir+caseName+'_epoch'+str(epoch)+'_hist'+str(history_length)+'_pred'+str(prediction_length)+'_AllrmseU.txt', np.array(AllrmseU))
        np.savetxt(testdir+caseName+'_epoch'+str(epoch)+'_hist'+str(history_length)+'_pred'+str(prediction_length)+'_maxrmseU.txt', np.array([maxrmseU]))
        
        np.savetxt(testdir+caseName+'_epoch'+str(epoch)+'_hist'+str(history_length)+'_pred'+str(prediction_length)+'_meanrmseU.txt', np.array([meanrmseU]))
        
        Alldata = np.array(Alldata)
        np.savez(testdir+caseName+'_epoch'+str(epoch)+'_samples'+str(i)+'_pred'+str(prediction_length)+'_hist'+str(history_length)+'_Alldata.txt',Alldata = Alldata)

        ## error for prediction mean
        err_mean = cal_err_mean(Alldata, ground_trueth)

        np.savetxt(testdir+caseName+'_epoch'+str(epoch)+'_hist'+str(history_length)+'_pred'+str(prediction_length)+'_err_mean.txt', err_mean)


        #interval_plot(modelref, x, ground_trueth, Alldata, epoch, prediction_length = prediction_length)
        #pdb.set_trace()


def cal_err_mean(ens_predict, ground_trueth, load_file = None):
    if load_file != None:
        ens_predict = np.load(load_file)['Alldata']
    mean_predict = np.mean(ens_predict, 0)
    mean_predict = torch.from_numpy(mean_predict).to('cuda:0')
    #pdb.set_trace()
    criterion  = torch.nn.MSELoss()
    Er=[]
    with torch.no_grad():
        for nC in range(mean_predict.shape[0]):
            print('Case ', str(nC))
            #pdb.set_trace()
            er = torch.sqrt(criterion(mean_predict[nC], ground_trueth[nC])/\
                                            criterion(ground_trueth[nC]*0, ground_trueth[nC]))
            Er.append(er.item())
    return Er
    #plt.figure()
    #plt.plot()
    #pdb.set_trace()




if __name__ == "__main__":
    # data = read_data(path="data/noise5.0_dataSave.txt", device='cuda:0')
    # data = data[:300].unsqueeze(0)
    # model = make_model(input_size=1024, num_layers=3, hidden_size=1024, num_cells=1024, dropout_rate=0.0, n_blocks=4,
    #                    n_hidden=3, prediction_step=99, mu_size=6, device='cuda:0')
    # model = load_model(model, 'saved_model/0/model_19500')
    # #x=data[:,0].unsqueeze(1)
    # x = data[:, :100]
    # ground=data[:,100:300]
    # test_model(model, x, 200, ground)

    caseName = dataset = 'cylinder'
    noise = 0.0
    filePath = "data/"+str(dataset)+"_noise"+str(noise)+"_dataSave.txt"
    dataAll = read_data(path=filePath, device = 'cuda:0')


    data = []
    mu_size = 50
    numT = 401
    #pdb.set_trace()
    for i in range(mu_size):     
    #for i in range(int(dataAll.shape[0]/numT)):
        tmp = dataAll[i*numT:(i+1)*numT].unsqueeze(0)
        data.append(tmp)

    data = torch.cat(data, dim = 0)
    #pdb.set_trace()
    print('size of data before is', data.shape)
    # ## pick every 2 cases
    # shape0 = 2*kwargs.mu_size
    # data = data[range(1,shape0,2),:]
    
    print('size of data is', data.shape)
    #pdb.set_trace()
    ## normalize data
    mean_d = data[:,:numT].mean(dim = (1,2), keepdim = True)
    std_d = data[:,:numT].std(dim = (1,2), keepdim = True)
    #pdb.set_trace()
    data = (data-mean_d)/std_d
    print('data is normalized')
    print('\n')

    nsamples = 20
    epoch = 80000
    history_length = 200
    prediction_length = 200


    modelref = 'transformer_Apr16_gluon_test_cylinder_normalized_cond4000_flow500_50cases_10pt_2nhidden_dm32_heads8_dim_scale1_nencode3_ndecode3_batch20'
    testdir = 'testres/'+str(modelref)+'/'

    endidx = min(data.shape[1], history_length+prediction_length)
    ground_trueth = data[:,history_length:endidx]
    load_file = testdir+caseName+'_epoch'+str(epoch)+'_samples'+str(nsamples-1)+'_pred'+str(prediction_length)+'_hist'+str(history_length)+'_Alldata.txt.npz'
    err_mean = cal_err_mean(None, ground_trueth, load_file = load_file)





