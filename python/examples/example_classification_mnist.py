import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pyrenn as prn

###
#Read Example Data
mnist = pickle.load( open( "MNIST_data.pkl", "rb" ) )
P = mnist['P']
Y = mnist['Y']
Ptest = mnist['Ptest']
Ytest = mnist['Ytest']


###
#Create and train NN

#create recurrent neural network with 28*28 inputs,
#1 hidden layers with 10 neurons 
#and 10 outputs (one for each possible class/number)
#the NN uses no delayed or recurrent inputs/connections
net = prn.CreateNN([28*28,10,10])

batch_size = 1000
number_of_batches=20

for i in range(number_of_batches):
    r = np.random.randint(0,25000-batch_size)
    Ptrain = P[:,r:r+batch_size]
    Ytrain = Y[:,r:r+batch_size]

    #Train NN with training data Ptrain=input and Ytrain=target
    #Set maximum number of iterations k_max
    #Set termination condition for Error E_stop
    #The Training will stop after k_max iterations or when the Error <=E_stop
    net = prn.train_LM(Ptrain,Ytrain,net,
                           verbose=True,k_max=1,E_stop=1e-5) 
    print('Batch No. ',i,' of ',number_of_batches)



###
#Select Test data

#Choose random number 0...5000-9
idx = np.random.randint(0,5000-9) 
#Select 9 random Test input data
P_ = Ptest[:,idx:idx+9]
#Calculate NN Output for the 9 random test inputs
Y_ = prn.NNOut(P_,net)


###
#PLOT
fig = plt.figure(figsize=[11,7])
gs = mpl.gridspec.GridSpec(3,3)

for i in range(9):
    
    ax = fig.add_subplot(gs[i])
    
    y_ = np.argmax(Y_[:,i]) #find index with highest value in NN output
    p_ = P_[:,i].reshape(28,28) #Convert input data for plotting
    
    ax.imshow(p_) #plot input data
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(str(y_), fontsize=18)
    
plt.show()
