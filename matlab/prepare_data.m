function [data,net] = prepare_data(P,Y,net,P0,Y0)
% 	Prepare Input Data for the use for NN Training and check for errors
% 		
% 	Args:
% 		P:		neural network Inputs
% 		Y: 		neural network Targets
% 		P0:		previous input data
% 		Y0:		previous output data
% 		net: 	neural network
% 	Returns:
% 		data:	struct containing data for training or calculating putput

if ~exist('P0', 'var')
    P0 = [];
end
if ~exist('Y0', 'var')
    Y0 = [];
end

%  Ceck if input and output data match structure of NN	
if size(P,1) ~= net.nn(1)
    error('Dimension of Input Data P does not match number of inputs of the NN')
end
if size(Y,1) ~= net.nn(end)
    error('Dimension of Output Data Y does not match number of outputs of the NN')
end
if size(Y,2) ~= size(P,2)
    error('Input P and output Y must have same number of datapoints Q')
end

%check if previous data is given
if (~isempty(P0)) && (~isempty(Y0))
    %  Ceck if input and output of prevoius data match structure of NN	
    if size(P0,1) ~= net.nn(1)
        error('Dimension of previous Input Data P0 does not match number of inputs of the NN')
    end
    if size(Y0,1) ~= net.nn(end)
        error('Dimension of previous Output Data Y0 does not match number of outputs of the NN')
    end
    if size(Y,2) ~= size(P,2)
    error('Previous Input and output data P0 and Y0 must have same number of datapoints Q0')
    end
    
    q0 = size(P0,2); %number of prevoius Datapoints given    
    a=cell(q0,net.M);  %initialise layer outputs
    for i=1:q0
        for j=1:net.M-1
            a{i,j}=zeros(net.nn(j+1),1); %layer ouputs of hidden layers are unknown -> set to zero
        end
        a{i,net.M}=Y0(:,i)./net.normY; %set layer ouputs of output layer 
    end
    %add previous inputs and outputs to inpu/output matrices
    P_ = [P0,P];
    Y_ = [Y0,Y];
else
	%add previous inputs and outputs to inpu/output matrices
    P_ = P;
    Y_ = Y;
    q0=0;
    a={};
end

% normalize
P_norm = P_;
Y_norm = Y_;
if isfield(net, 'normP')==0
    normP = ones(size(P_,1),1);
    for p = 1:size(P_,1)
        normP(p) = max(max(abs(P_(p,:))),1.0);
        P_norm(p,:) = P_(p,:)./normP(p);
    end
    normY = ones(size(Y_,1),1);
    for y = 1:size(Y_,1)
        normY(y) = max(max(abs(Y_(y,:))),1.0);
        Y_norm(y,:) = Y_(y,:)./normY(y);
    end
    net.normP = normP;
    net.normY = normY;
else
    for p = 1:size(P_,1)
        P_norm(p,:) = P_(p,:)./net.normP(p);
    end
    for y = 1:size(Y_,1)
        Y_norm(y,:) = Y_(y,:)./net.normY(y);
    end
end

%Create data dict
data = {};		
data.P = P_norm;
data.Y = Y_norm;
data.a = a;
data.q0 = q0;

end
