'''
This example shows how to calculate the gradient vector or derivative vector
of the Error of a NN with respect to the weight vector
The gradient vector can be calculated either using RTRL or BPTT algorithm
'''
import numpy as np
import pandas as pd
import time

import pyrenn as prn

###
#Read Example Data
df = pd.ExcelFile('example_data.xlsx').parse('pt2')
P = df['P'].values
Y = df['Y'].values

###
#Create and train NN

#create recurrent neural network with 1 input, 2 hidden layers with 
#2 neurons each and 1 output
#the NN has a recurrent connection with delay of 1 timestep in the hidden
# layers and a recurrent connection with delay of 1 and 2 timesteps from the output
# to the first layer
net = prn.CreateNN([1,2,2,1],dIn=[0],dIntern=[1],dOut=[1,2])

###
#Prepare input Data for gradient calculation
data,net = prn.prepare_data(P,Y,net)

###
#Calculate derivative vector (gradient vector)

#Real Time Recurrent Learning
t0_rtrl = time.time()
J,E,e = prn.RTRL(net,data)
g_rtrl = 2 * np.dot(J.transpose(),e) #calculate g from Jacobian and error vector
t1_rtrl = time.time()

#Back Propagation Through Time
t0_bptt = time.time()
g_bptt,E = prn.BPTT(net,data)
t1_bptt = time.time()


###
#Compare
print('\n\n\nComparing Methods:') 
print('Time RTRL: ',(t1_rtrl-t0_rtrl),'s')
print('Time BPTT: ',(t1_bptt-t0_bptt),'s')
if not np.any(np.abs(g_rtrl-g_bptt)>1e-9):
	print('\nBoth methods showing the same result!')
	print('g_rtrl/g_bptt = ',g_rtrl/g_bptt)
