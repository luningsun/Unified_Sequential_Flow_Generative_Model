import numpy as np
from matplotlib import pyplot as plt
import pdb
testdir = 'testres/transformer_Apr16_gluon_test_cylinder_normalized_cond4000_flow500_50cases_10pt_2nhidden_dm32_heads8_dim_scale1_nencode3_ndecode3_batch20/'
caseName = 'cylinder'
epoch = 80000
history_length = 200
prediction_length = 200
AllrmseU = np.loadtxt(testdir+caseName+'_epoch'+str(epoch)+'_hist'+str(history_length)+'_pred'+str(prediction_length)+'_AllrmseU.txt')
#pdb.set_trace()
plt.figure()
for i in range(AllrmseU.shape[0]):
    plt.plot(AllrmseU[i,:],'x')
    plt.title('epoch '+str(epoch))

    plt.axis([-0.5, 50.5, 0, 0.72])
plt.savefig('testrmse_epoch'+str(epoch)+'.png', bbox_inches = 'tight')
plt.show()