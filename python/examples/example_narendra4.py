import pandas as pd
import matplotlib.pyplot as plt

import pyrenn as prn

###
#Read Example Data
df = pd.ExcelFile('example_data.xlsx').parse('narendra4')
P = df['P'].values
Y = df['Y'].values
Ptest = df['Ptest'].values
Ytest = df['Ytest'].values


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
prn.train_LM(P,Y,net,verbose=True,k_max=200,E_stop=1e-3) 


###
#Calculate outputs of the trained NN for train and test data
y = prn.NNOut(P,net)
ytest = prn.NNOut(Ptest,net)

###
#Plot results
fig = plt.figure(figsize=(11,7))
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
fs=18

#Train Data
ax0.set_title('Train Data',fontsize=fs)
ax0.plot(y,color='b',lw=2,label='NN Output')
ax0.plot(Y,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Train Data')
ax0.tick_params(labelsize=fs-2)
ax0.legend(fontsize=fs-2,loc='upper left')
ax0.grid()

#Test Data
ax1.set_title('Test Data',fontsize=fs)
ax1.plot(ytest,color='b',lw=2,label='NN Output')
ax1.plot(Ytest,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
ax1.tick_params(labelsize=fs-2)
ax1.legend(fontsize=fs-2,loc='upper left')
ax1.grid()

fig.tight_layout()
plt.show()
