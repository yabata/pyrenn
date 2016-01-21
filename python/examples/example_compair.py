import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pyrenn as prn

###
#Read Example Data
df = pd.ExcelFile('example_data.xlsx').parse('compressed_air')
P = np.array([df['P1'].values,df['P2'].values,df['P3'].values])
Y = np.array([df['Y1'].values,df['Y2']])
Ptest = np.array([df['P1test'].values,df['P2test'].values,df['P3test'].values])
Ytest = np.array([df['Y1test'].values,df['Y2test']])


###
#Create and train NN

#create feed forward neural network with 1 input, 2 hidden layers with 
#4 neurons each and 1 output
#the NN has a recurrent connection with delay of 1 timesteps from the output
# to the first layer
net = prn.CreateNN([3,5,5,2],dIn=[0],dIntern=[],dOut=[1])

#Train NN with training data P=input and Y=target
#Set maximum number of iterations k_max to 500
#Set termination condition for Error E_stop to 1e-5
#The Training will stop after 500 iterations or when the Error <=E_stop
net = prn.train_LM(P,Y,net,verbose=True,k_max=500,E_stop=1e-5) 


###
#Calculate outputs of the trained NN for train and test data
y = prn.NNOut(P,net)
ytest = prn.NNOut(Ptest,net)

###
#Plot results
fig = plt.figure(figsize=(15,10))
ax0 = fig.add_subplot(221)
ax1 = fig.add_subplot(222,sharey=ax0)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224,sharey=ax2)
fs=18

t = np.arange(0,480.0)/4 #480 timesteps in 15 Minute resolution
#Train Data
ax0.set_title('Train Data',fontsize=fs)
ax0.plot(t,y[0],color='b',lw=2,label='NN Output')
ax0.plot(t,Y[0],color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Data')
ax0.tick_params(labelsize=fs-2)
ax0.legend(fontsize=fs-2,loc='upper left')
ax0.grid()
ax0.set_ylabel('Storage Pressure [bar]',fontsize=fs)
plt.setp(ax0.get_xticklabels(), visible=False)

ax2.plot(t,y[1],color='b',lw=2,label='NN Output')
ax2.plot(t,Y[1],color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Train Data')
ax2.tick_params(labelsize=fs-2)
ax2.grid()
ax2.set_xlabel('Time [h]',fontsize=fs)
ax2.set_ylabel('el. Power [kW]',fontsize=fs)

#Test Data
ax1.set_title('Test Data',fontsize=fs)
ax1.plot(t,ytest[0],color='b',lw=2,label='NN Output')
ax1.plot(t,Ytest[0],color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax1.tick_params(labelsize=fs-2)
# ax1.legend(fontsize=fs-2,loc='upper left')
ax1.grid()
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), visible=False)

ax3.plot(t,ytest[1],color='b',lw=2,label='NN Output')
ax3.plot(t,Ytest[1],color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax3.tick_params(labelsize=fs-2)
ax3.grid()
ax3.set_xlabel('Time [h]',fontsize=fs)
plt.setp(ax3.get_yticklabels(), visible=False)

fig.tight_layout()
plt.show()
