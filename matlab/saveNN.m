function saveNN(net,filename)
    %Save neural network object to file
    % 		
    % 	Args:
    % 		net:        neural network object
    % 		filename:	path of csv file to save neural network

    %write neural network structure nn
    fid = fopen(filename, 'w') ;
        fprintf(fid, '%s\n', 'nn') ;
    fclose(fid) ;
    dlmwrite(filename, net.nn,'-append');

    %write input delays dIn
    fid = fopen(filename, 'a') ;
    fprintf(fid, '%s\n', 'dIn') ;
    fclose(fid) ;
    dlmwrite(filename, net.delay.In,'-append');


    %write iternal delays dIntern
    fid = fopen(filename, 'a') ;
    fprintf(fid, '%s\n', 'dIntern') ;
        if isempty(net.delay.Intern)
             fprintf(fid, ',\n') ;
        end
    fclose(fid) ;
    dlmwrite(filename, net.delay.Intern,'-append');

    %write output delays dOut
    fid = fopen(filename, 'a') ;
    fprintf(fid, '%s\n', 'dOut') ;
        if isempty(net.delay.Out)
             fprintf(fid, ',\n') ;
        end
    fclose(fid) ;
    dlmwrite(filename, net.delay.Out,'-append');

    %write factor for input data normalization normP
    fid = fopen(filename, 'a') ;
    fprintf(fid, '%s\n', 'normP') ;
    fclose(fid) ;
    dlmwrite(filename, net.normP','precision','%.17f','-append');

    %write factor for output data normalization normY
    fid = fopen(filename, 'a') ;
    fprintf(fid, '%s\n', 'normY') ;
    fclose(fid) ;
    dlmwrite(filename, net.normY','precision','%.17f','-append');

    %write fweight vector w
    fid = fopen(filename, 'a') ;
    fprintf(fid, '%s\n', 'w') ;
    fclose(fid) ;
    dlmwrite(filename, net.w,'precision','%.17f','-append'); 
end