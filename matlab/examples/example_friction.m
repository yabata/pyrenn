%%
%Read Example Data
file = 'example_data.xlsx';
num = xlsread(file,'friction');
%Inputs and outputs have to be matrices where columns=datapoints
%and rows=inputs
P = num(1:41,2).';
Y = num(1:41,3).';
Ptest = num(:,4).';
Ytest = num(:,5).';

%%
%Create NN

%create feed forward neural network with 1 input, 2 hidden layers with 
%3 neurons each and 1 output
net = CreateNN([1 3 3 1]); %alternative: net = CreateNN([1,3,3,1],[0],[],[]);

%%
%Train with LM-Algorithm
% Train NN with training data P=input and Y=target
% Set maximum number of iterations k_max to 100
% Set termination condition for Error E_stop to 1e-5
% The Training will stop after 100 iterations or when the Error <=E_stop
netLM = train_LM(P,Y,net,100,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
y_LM = NNOut(P,netLM); 
ytest_LM = NNOut(Ptest,netLM); 

%%
%Train with BFGS-Algorithm
% Train NN with training data P=input and Y=target
% Set maximum number of iterations k_max to 200
% Set termination condition for Error E_stop to 1e-5
% The Training will stop after 200 iterations or when the Error <=E_stop
% measure time dt
netBFGS = train_BFGS(P,Y,net,200,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
y_BFGS = NNOut(P,netBFGS); 
ytest_BFGS = NNOut(Ptest,netBFGS); 


%%
%Plot Results
fig = figure();
set(fig, 'Units', 'normalized', 'Position', [0.2, 0.1, 0.6, 0.6])
axis tight

subplot(311)
set(gca,'FontSize',16)
plot(P,Y,'r:','LineWidth',2)
hold on
grid on
plot(P,y_LM,'b','LineWidth',2)
plot(P,y_BFGS,'g','LineWidth',2)
l1 = legend('Train Data','LM output','BFGS output','Location','northwest');
set(l1,'FontSize',14)

subplot(312)
set(gca,'FontSize',16)
plot(Ptest,Ytest,'r:','LineWidth',2)
hold on
grid on
plot(Ptest,ytest_LM,'b','LineWidth',2)
plot(Ptest,ytest_BFGS,'g','LineWidth',2)
l2 = legend('Test Data','LM output','BFGS output','Location','northwest');
set(l2,'FontSize',14)

subplot(313)
set(gca,'FontSize',16)
plot(netLM.ErrorHistory,'b','LineWidth',2)
hold on
grid on
plot(netBFGS.ErrorHistory,'g','LineWidth',2)
ylim([0,0.1])
l3 = legend('LM Error','BFGS Error','Location','northeast');
set(l3,'FontSize',14)

