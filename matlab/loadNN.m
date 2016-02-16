function net = loadNN(filename)
    %Load neural network object from file
    % 		
    % 	Args:
    % 		filename:	path to csv file to save neural network
    %
    %   Returns:
    %       net:        neural network object

    fid = fopen(filename);

    %read neural network structure nn
    temp = fgetl(fid);
    nn = str2num(fgetl(fid));

    %read input delays dIn
    temp = fgetl(fid);
    temp = fgetl(fid);
    if isnumeric(temp)
       dIn = temp;
    else
       dIn = str2num(temp);
    end
    
    %read iternal delays dIntern
    temp = fgetl(fid);
    temp = fgetl(fid);
    if isnumeric(temp)
        dIntern = temp;
    elseif temp==','
        dIntern = []; 
    else
        dIntern = str2num(temp);
    end
   
    %read output delays dOut
    temp = fgetl(fid);
    temp = fgetl(fid);
    if isnumeric(temp)
        dOut = temp;
    elseif temp==','
        dOut = []; 
    else
        dOut = str2num(temp);
    end
    
    %read factor for input data normalization normP
    temp = fgetl(fid);
    temp = fgetl(fid);
    if isnumeric(temp)
       normP = temp;
    else
       normP = str2num(temp)';
    end
    
    %read factor for output data normalization normY
    temp = fgetl(fid);
    temp = fgetl(fid);
    if isnumeric(temp)
       normY = temp;
    else
       normY = str2num(temp)';
    end
    fclose(fid);
    
    %read weight vector w
    w = csvread(filename,13,0);
    
    %Create neural network and assign loaded weights and factors
    net = CreateNN(nn,dIn,dIntern,dOut);
    net.normP = normP;
    net.normY = normY;
    net.w = w;
    
    
end