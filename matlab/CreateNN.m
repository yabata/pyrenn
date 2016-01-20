% Netzaufbau mit Gewichten und Verzögerungen erstellen und wichtige ngen
% erstellen

function [net]=CreateNN(nn,dIn,dIntern,dOut)

% dIn: Zelle für jeden Eingang mit Vektor der Verzögerungen
if ~exist('dIn', 'var')
    dIn = [0];
end
% dIntern: Vektor mit den Verzögerungen der Internen Schichten (AußerSM -> S1)
if ~exist('dIntern', 'var')
    dIntern = [];
end
% dOut: Zelle für jeden Ausgang mit Vektor der Verzögerungen zur Schicht S1
if ~exist('dOut', 'var')
    dOut = [];
end





% NN=[P S1 S2 .. SM] Aufbau des Netzes( P steht für Anzahl der
    % Eingänge, Sm für die Anzahl der ronen von Schicht m

    
  
net.nn=nn; %Gesamtaufbau mit Eingängen
net.delay.In=dIn;    %Eingansverzögerung
net.delay.Intern=dIntern;    %Interne Verzögerungen
net.delay.Out=dOut;    %Ausgangsverzögerungen

net.M=length(nn)-1;        %Anzahl der Schichten des Neuronalen Netzes
net.layers=nn(2:end);      % Aufbau des NN ohne Eingangsmatrix [S1 S2...SM]
net.dmax=max([net.delay.In,net.delay.Intern,net.delay.Out]); %Maximale Verzögerung innerhalb des NN


[net]=w_Create(net); %Gesamtgewichtsvektor Net.w_0 erzeugen sowie wichtige Mengen erzeugen

net.N=length(net.w_0); %Anzahl der Gewichte
net.w=net.w_0; %Zu Otimierender Gesamtgewichtsvektor Net.w_k ist zu Beginn der Startvektor Net.w_0

end