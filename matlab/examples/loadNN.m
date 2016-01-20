function net = loadNN(filename)
    %Load neural network object from file
    % 		
    % 	Args:
    % 		filename:	path to csv file to save neural network
    %
    %   Returns:
    %       net:        neural network object

    [~,~,rawData] = xlsread(filename);

    %read neural network structure nn
    nn = str2num(rawData{2,1});

    %read input delays dIn
    if isnumeric(rawData{4,1})
       dIn = rawData{4,1};
    else
       dIn = str2num(rawData{4,1});
    end
    
    %read iternal delays dIntern
    if isnumeric(rawData{6,1})
        dIntern = rawData{6,1};
    elseif rawData{6,1}==','
        dIntern = []; 
    else
        dIntern = str2num(rawData{6,1});
    end
   
    %read output delays dOut
    if isnumeric(rawData{8,1})
        dOut = rawData{8,1};
    elseif rawData{8,1}==','
        dOut = []; 
    else
        dOut = str2num(rawData{8,1});
    end
    
    %read factor for input data normalization normP
    if isnumeric(rawData{10,1})
       normP = rawData{10,1};
    else
       normP = str2num(rawData{10,1})';
    end
    
    %read factor for output data normalization normY
    if isnumeric(rawData{12,1})
       normY = rawData{12,1};
    else
       normY = str2num(rawData{12,1})';
    end
    
    %read weight vector w
    w = csvread(filename,13,0);
    
    %Create neural network and assign loaded weights and factors
    net = CreateNN(nn,dIn,dIntern,dOut);
    net.normP = normP;
    net.normY = normY;
    net.w = w;
    
    
end