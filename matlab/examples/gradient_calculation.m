%%
%Read Example Data
file = 'example_data.xlsx';
num = xlsread(file,'pt2');
%Inputs and outputs have to be matrices where columns=datapoints
%and rows=inputs
P = num(:,2).';
Y = num(:,3).';

%%
%Create NN

% create recurrent neural network with 1 input, 2 hidden layers with 
% 2 neurons each and 1 output
% the NN has a recurrent connection with delay of 1 timestep in the hidden
% layers and a recurrent connection with delay of 1 and 2 timesteps from the output
% to the first layer
nn = [1 2 2 1];
dIn = [0];
dIntern=[1];
dOut=[1,2];
net = CreateNN(nn,dIn,dIntern,dOut); %alternative: net = CreateNN([1,2,2,1],[0],[1],[1,2]);

%%
%Prepare input Data for gradient calculation
data = prepare_data(P,Y,net,{},0);

%%
%Calculate derivative vector (gradient vector)

%Real Time Recurrent Learning
t0_rtrl = cputime;
[J,E,e] = RTRL(net,data);
g_rtrl = 2.*J'*e;
t1_rtrl = cputime;

%Back Propagation Through Time
t0_bptt = cputime;
g_bptt = BPTT(net,data);
t1_bptt = cputime;

%%
%Compare
disp(' ')
disp(['Comparing Methods:'])
disp(['Time RTRL: ',num2str(t1_rtrl-t0_rtrl),'s'])
disp(['Time BPTT: ',num2str(t1_bptt-t0_bptt),'s'])
if ~any(abs(g_rtrl-g_bptt)>1e-9)
	disp(sprintf('\nBoth methods showing the same result!'))
    disp('g_rtrl/g_bptt = ')
    (g_rtrl./g_bptt)'
end