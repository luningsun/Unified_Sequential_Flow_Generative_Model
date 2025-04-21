import numpy as np
import pdb
import random

from matplotlib import pyplot as plt
seed = 1


random.seed(seed)
np.random.seed(seed)
noise = 0.
N = 100 # number of samples
L = 1000 # length of each sample (number of values for each sine wave)
T = 60# width of the wave
x = np.empty((N,L), np.float32) # instantiate empty array
#x[:] = np.arange(L) + np.random.randint(-4*T, 4*T, N).reshape(N,1)

x = np.linspace(0,T,L) + np.zeros([N,L])
#pdb.set_trace()
##
## random sample from 1 to 2
A = np.random.randn(N, 1)+1
#A = 1
## random sample from 0 to pi
B = np.pi*np.random.randn(N, 1)
# random sample from -pi/2, pi/2

C = np.pi*np.random.randn(N, 1)-np.pi/2
#y = np.sin(x/1.0/T).astype(np.float32)
y = A*np.sin(B*x+C)
plt.figure()
for i in range(0,N,1):
    plt.plot(x[i], y[i], label = str(i))
#plt.legend()
plt.savefig('3dofParaSine.png', bbox_inches = 'tight')

fname = "data/sine_noise"+str(str(noise))+"_dataSave.txt"

twod_mat = np.array([]).reshape(0,1)

for i in range(N):
    twod_mat = np.vstack([twod_mat, y[i].reshape(-1,1)])

np.savetxt(fname,twod_mat)
pdb.set_trace()