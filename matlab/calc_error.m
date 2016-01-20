% Calculate Error for NN based on data
% 
% Args:
%     net:	neural network
%     data: 	Training Data
% Returns:
%     E:		Mean squared Error of the Neural Network compared to Training data

function [E] = calc_error(net,data) 


P=data.P;   %Eingangsdaten
Y=data.Y;   %Ausgangsdaten des Systems
a=data.a;   %Schichtausgänge
q0=data.q0; %Ab q0.tem Trainingsdatum Ausgänge berechnen


M = net.M;      %Anzahl der Schichten des NN
layers=net.layers;    %Aufbau des Netzes
dI=net.dI;      %Verzögerung der Eingänge
dL=net.dL;      %Verzögerung zwischen den Schichten
L_f=net.L_f;    %Vorwärtsverbindungen der Schichten

I=net.I;        %Eingänge in die Schichten


% Gesamtgewichtvektor in Gewichtmatrizen/ Vektoren umwandeln
[IW,LW,b]=w2Wb(net);

%% 1. Vorwärtspropagieren:
[Y_NN,n,a] = NNOut_(P,net,IW,LW,b,a,q0);     %Ausgänge, Summenausgänge und Schichtausgänge des NN berechnen


%% 2. Kostenfunktion berechnen:
Y_delta=Y-Y_NN;   %Fehlermatrix: Fehler des NN-Ausgangs bezüglich des Systemausgangs für jeden Datenpunkt
e=reshape(Y_delta,[],1);    %Fehlervektor (untereinander) [y_delta1_ZP1;y_delta2_ZP1;...;y_delta1_ZPX;y_delta2_ZPX]
E=e'*e; %Kostenfunktion: Summierter Quadratischer Fehler