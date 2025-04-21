import numpy as np
from matplotlib import pyplot as plt
import pdb

lengthT = 100
total_Pt = 1024
res = []
for i in range(lengthT):
    data = np.load('tentative_res/pred_time'+str(i)+'_mean_sigma.npz')
    #pdb.set_trace()   
    print('sigma mean is', data['base_dist_sigma'].mean())
    print('sigma is', data['base_dist_sigma'])
    #pdb.set_trace()
    res.append(data['base_dist_sigma'][0])
    #pdb.set_trace()
    #base_dist_var = data['base_dist_var']
res = np.concatenate(res, axis = 0)
#pdb.set_trace()


for k in range(total_Pt):
    plt.figure()
    plt.plot(res[:, k])
    plt.xlabel('t')
    plt.ylabel('sigma')
    plt.title('Point'+str(k))
    #pdb.set_trace()
    plt.savefig('tentative_res/plot/varPt'+str(k)+'.png', bbox_inches = 'tight')
    plt.close('all')
    
