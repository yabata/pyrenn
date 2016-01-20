import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyrenn as prn

###
#Read Example Data
df = pd.ExcelFile('example_data.xlsx').parse('narendra4')
P = df['P'].values
Y = df['Y'].values
Ptest_ = df['Ptest'].values
Ytest_ = df['Ytest'].values

#define the first 3 timesteps t=[0,1,2] of Test Data as previous (known) data P0test and Y0test
P0test = Ptest_[0:3]
Y0test = Ytest_[0:3]
#Use the timesteps t = [3..99] as Test Data
Ptest = Ptest_[3:100]
Ytest = Ytest_[3:100]

###
#Create and train NN

#create recurrent neural network with 1 input, 2 hidden layers with 
#3 neurons each and 1 output
#the NN uses the input data at timestep t-1 and t-2
#The NN has a recurrent connection with delay of 1,2 and 3 timesteps from the output
# to the first layer (and no recurrent connection of the hidden layers)
net = prn.CreateNN([1,3,3,1],dIn=[1,2],dIntern=[],dOut=[1,2,3])

#Train NN with training data P=input and Y=target
#Set maximum number of iterations k_max to 200
#Set termination condition for Error E_stop to 1e-3
#The Training will stop after 200 iterations or when the Error <=E_stop
net = prn.train_LM(P,Y,net,verbose=True,k_max=200,E_stop=1e-3) 


###
#Calculate outputs of the trained NN for test data with and without previous input P0 and output Y0
ytest = prn.NNOut(Ptest,net)
y0test = prn.NNOut(Ptest,net,P0=P0test,Y0=Y0test)

###
#Plot results
fig = plt.figure(figsize=(11,7))
ax1 = fig.add_subplot(111)
fs=18

#Test Data
ax1.set_title('Test Data',fontsize=fs)
ax1.plot(ytest,color='b',lw=2,label='NN Output without P0,Y0')
ax1.plot(y0test,color='g',lw=2,label='NN Output with P0,Y0')
ax1.plot(Ytest,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax1.tick_params(labelsize=fs-2)
ax1.legend(fontsize=fs-2,loc='lower right')
ax1.grid()

fig.tight_layout()
plt.show()
