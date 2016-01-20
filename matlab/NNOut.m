% Berechnung des Ausgangs Y_NN des Neuronalen Netzes Net mit dem
% Gewichtsvektor Net.w_k und den Eingangsdaten Data.P

function [Y_NN_scaled] = NNOut(P,net,P0,Y0)

if ~exist('P0', 'var')
    P0 = [];
end
if ~exist('Y0', 'var')
    Y0 = [];
end

Y = zeros(net.layers(end),size(P,2));
[data,net] = prepare_data(P,Y,net,P0,Y0);
[IW,LW,b]=w2Wb(net); % Gesamtgewichtvektor in Gewichtmatrizen/ Vektoren umwandeln

%% Vorwärtspropagieren:
[Y_NN] = NNOut_(data.P,net,IW,LW,b,data.a,data.q0) ;     %Ausgänge des NN berechnen

Y_NN_scaled = Y_NN;
    for y = 1:size(Y_NN_scaled,1)
        Y_NN_scaled(y,:) = Y_NN_scaled(y,:).*net.normY(y);
    end

end