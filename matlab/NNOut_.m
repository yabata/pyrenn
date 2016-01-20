% Berechnung des Ausgangs Y_NN, der Summenausgänge n und der Schichtausgänge a 
% des Neuronalen Netzes Net mit den
%   Eingangsdaten P
%   (alten) Schichtausgängen a
%   Start Trainingsdatum zur Berechnung der Ausgänge q0
%   Eingangssgewichtsmatrizen IW und den Eingangsverzögerungen dI
%   Eingänge in die Schichten I
%   Verbindungsgewichtsmatrizen LW und den Verzögerung zwischen den Schichten dL
%   Vorwärtsverbindungen der Schichten L_f
%   Biasgewichten b
%   Aufbau des Netzes net
%   Anzahl der Schichten des NN M

function [Y_NN,n,a] = NNOut_(P,net,IW,LW,b,a,q0) 

dL = net.dL; %dL[m,l]: delays for the connection layer m -> layer l
dI = net.dI; %Delays for connection of input to layer 1
I = net.I; %set of inputs with a connection to layer 1
L_f = net.L_f; % L_f[m]: set of layers with a forward connection to layer m
M = net.M; %number of layers of NN
outputs = net.nn(end); %number of outputs

Q = size(P,2);  %Anzahl der Datenpunkte
n=cell(Q,M);    %Definition von n: Summenausgang der Schichten


Y_NN=zeros(outputs,Q); % Ausgangsmatrix des NN

for q=q0+1:Q
    
    a{q,1}=0;
    for m=1:M    %für alle Schichten m
        n{q,m}=0;   % Summenausgang Datenpunkt q, Schicht m
        %--------------------
        %Eingangsgewichte
        if m==1
            for i=I{m}  %Alle Eingänge i
                for d=dI{m,i}   % alle Verzögerungen i->m
                    if (q-d)>0
                        n{q,m}=n{q,m}+IW{m,i,d+1}*P(:,q-d);   %Zu Summenausgang Schicht m addieren
                    end
                end
            end
        end
        %-----------------------    
       %Verbindungsgewichte
        for l=L_f{m}  %Alle Schichten l die eine Vorwärtsverbindung zur Schicht m besitzen
            for d=dL{m,l}   % alle Verzögerungen l->m
                if (q-d)>0
                    n{q,m}=n{q,m}+LW{m,l,d+1}*a{q-d,l};   %Zu Summenausgang Schicht m addieren
                end
            end
        end
        %-------------------------
        %Biasgewichte
        n{q,m}=n{q,m}+b{m};     %Bias zuu Summenausgang Schicht m addieren
                
        %-------------
        %Schichtausgang berechnen
        if m==M
            a{q,M}=n{q,M};       %Ausgang der letzten Schicht (keine Funktion mehr), Datensatz q
        else
            a{q,m}=tanh(n{q,m});                %Funktionsgleichung Schicht m, Datensatz q
        end
    end    
    Y_NN(:,q)=a{q,M};                %Matrix der Netzausgangsdaten, jeder Ausgang ist ein Vektor (untereinander) [y_1;y_2;...; y_SM]   
end
Y_NN=Y_NN(:,q0+1:end);