%%
%Read Example Data
file = 'example_data.xlsx';
num = xlsread(file,'narendra4');
%Inputs and outputs have to be matrices where columns=datapoints
%and rows=inputs
P = num(:,2).';
Y = num(:,3).';
Ptest_ = num(:,4).';
Ytest_ = num(:,5).';

%define the first 3 timestep t=[1,2,3] of Test Data as previous (known)
%data P0test and Y0test
P0test = Ptest_(:,1:3);
Y0test = Ytest_(:,1:3);
%Use the timesteps t = [4..100] as Test Data
Ptest = Ptest_(:,4:100);
Ytest = Ytest_(:,4:100);

%%
%Create NN

%create recurrent neural network with 1 input, 2 hidden layers with 
%3 neurons each and 1 output
%the NN uses the input data at timestep t-1 and t-2
%The NN has a recurrent connection with delay of 1,2 and 3 timesteps from the output
% to the first layer (and no recurrent connection of the hidden layers)
nn = [1 3 3 1];
dIn = [1,2];
dIntern=[];
dOut=[1,2,3];
net = CreateNN(nn,dIn,dIntern,dOut); %alternative: net = CreateNN([1,3,3,1],[1,2],[],[1,2,3]);

%%
%Train with LM-Algorithm
% Train NN with training data P=input and Y=target
% Set maximum number of iterations k_max to 200
% Set termination condition for Error E_stop to 1e-3
% The Training will stop after 200 iterations or when the Error <=E_stop
net = train_LM(P,Y,net,200,1e-3);


%Calculate Output of trained net (LM) for and Test with and without P0 and Y0
ytest = NNOut(Ptest,net); 
y0test = NNOut(Ptest,net,P0test,Y0test); 


%%
%Plot Results
fig = figure();
set(fig, 'Units', 'normalized', 'Position', [0.2, 0.1, 0.6, 0.6])
axis tight

subplot(111)
set(gca,'FontSize',16)
set(gca, 'LooseInset', get(gca,'TightInset'))
plot(Ytest,'r:','LineWidth',2)
hold on
grid on
plot(ytest,'b','LineWidth',2)
plot(y0test,'g','LineWidth',2)
l2 = legend('Test Data','NN output without P0,Y0','NN Output with P0,Y0','Location','northoutside','Orientation','horizontal');
set(l2,'FontSize',14)
