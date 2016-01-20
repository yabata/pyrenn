import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyrenn as prn

###
#Read Example Data
df = pd.ExcelFile('example_data.xlsx').parse('compressed_air')
P = np.array([df['P1'].values,df['P2'].values,df['P3'].values])
Y = np.array([df['Y1'].values,df['Y2']])
Ptest_ = np.array([df['P1test'].values,df['P2test'].values,df['P3test'].values])
Ytest_ = np.array([df['Y1test'].values,df['Y2test']])

#define the first timestep t=0 of Test Data as previous (known) data P0test and Y0test
P0test = Ptest_[:,0:1]
Y0test = Ytest_[:,0:1]
#Use the timesteps t = [1..99] as Test Data
Ptest = Ptest_[:,1:100]
Ytest = Ytest_[:,1:100]

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
prn.train_LM(P,Y,net,verbose=True,k_max=500,E_stop=1e-5) 


###
#Calculate outputs of the trained NN for test data with and without previous input P0 and output Y0
ytest = prn.NNOut(Ptest,net)
y0test = prn.NNOut(Ptest,net,P0=P0test,Y0=Y0test)


###
#Plot results
fig = plt.figure(figsize=(15,10))
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212,sharey=ax0)
fs=18

t = np.arange(0,np.shape(Ptest)[1])/4.0 #timesteps in 15 Minute resolution

#Test Data
ax0.set_title('Test Data',fontsize=fs)
ax0.plot(t,ytest[0],color='b',lw=2,label='NN Output without P0,Y0')
ax0.plot(t,y0test[0],color='g',lw=2,label='NN Output with P0,Y0')
ax0.plot(t,Ytest[0],color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax0.tick_params(labelsize=fs-2)
ax0.legend(fontsize=fs-2,loc='upper right')
ax0.grid()
plt.setp(ax0.get_xticklabels(), visible=False)
ax0.set_ylabel('Storage Pressure [bar]',fontsize=fs)

ax1.plot(t,ytest[1],color='b',lw=2,label='NN Output without P0,Y0')
ax1.plot(t,y0test[1],color='g',lw=2,label='NN Output with P0,Y0')
ax1.plot(t,Ytest[1],color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax1.tick_params(labelsize=fs-2)
ax1.grid()
ax1.set_xlabel('Time [h]',fontsize=fs)
ax1.set_ylabel('el. Power [kW]',fontsize=fs)

fig.tight_layout()
plt.show()
