%%
%Read Example Data
file = 'example_data.xlsx';
num = xlsread(file,'compressed_air');
%Inputs and outputs have to be matrices where columns=datapoints
%and rows=inputs
P = num(:,2:4).';
Y = num(:,5:6).';
Ptest_ = num(:,7:9).';
Ytest_ = num(:,10:11).';

%define the first timestep t=1 of Test Data as previous (known)
%data P0test and Y0test
P0test = Ptest_(:,1);
Y0test = Ytest_(:,1);
%Use the timesteps t = [2..100] as Test Data
Ptest = Ptest_(:,2:100);
Ytest = Ytest_(:,2:100);

%%
%Create NN

%create recurrent neural network with 3 inputs, 2 hidden layers with 
%5 neurons each and 2 outputs
%the NN uses the input data at timestep t-1 and t-2
%The NN has a recurrent connection with delay of 1,2 and 3 timesteps from the output
% to the first layer (and no recurrent connection of the hidden layers)
nn = [3 5 5 2];
dIn = [0];
dIntern=[];
dOut=[1];
net = CreateNN(nn,dIn,dIntern,dOut); %alternative: net = CreateNN([3,5,5,2],[0],[],[1]);

%%
%Train with LM-Algorithm
% Train NN with training data P=input and Y=target
% Set maximum number of iterations k_max to 500
% Set termination condition for Error E_stop to 1e-5
% The Training will stop after 500 iterations or when the Error <=E_stop
net = train_LM(P,Y,net,500,1e-5);


%Calculate Output of trained net (LM) for and Test with and without P0 and Y0
ytest = NNOut(Ptest,net); 
y0test = NNOut(Ptest,net,P0test,Y0test); 

%%
%Plot Results
t = (1:size(Ptest,2))./4; %timesteps in 15 Minute resolution

fig = figure();
set(fig, 'Units', 'normalized', 'Position', [0.2, 0.1, 0.6, 0.6])

subplot(211)
title('Train Data')
set(gca,'FontSize',16)
plot(t,Ytest(1,:),'r:','LineWidth',2)
hold on
grid on
plot(t,ytest(1,:),'b','LineWidth',2)
plot(t,y0test(1,:),'g','LineWidth',2)
l2 = legend('Test Data','NN output without P0,Y0','NN Output with P0,Y0','Location','northoutside','Orientation','horizontal');
set(l2,'FontSize',14)
set(gca, 'LooseInset', get(gca,'TightInset'))
axis tight
ylabel('Storage Pressure [bar]')


subplot(212)
set(gca,'FontSize',16)
plot(t,Ytest(2,:),'r:','LineWidth',2)
hold on
grid on
plot(t,ytest(2,:),'b','LineWidth',2)
plot(t,y0test(2,:),'g','LineWidth',2)
xlabel('time [h]')
set(gca, 'LooseInset', get(gca,'TightInset'))
axis tight
ylabel('el. Power [kW]')
