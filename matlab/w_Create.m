% Eingangs- und Verbindungsmatrizen sowie Biasvektoren für in Net
% vorgegebene Netzstruktur ertsellen und in Gesamtgewichtsvektor Net.w_0
% zusammenfassen
% Zusätzlich werden verschiedenen wichtige Mengen definiert
function [net]=w_Create(net)

%Zufallszahlen zufäälig machen
ms_time=str2num(datestr(now,'FFF'));
RStr = RandStream('mcg16807','Seed',ms_time);
RandStream.setGlobalStream(RStr);


M=net.M;    %Anzahl der Schichten des NN
layers=net.layers; % Aufbau des NN
inputs=net.nn(1); %Anzahl 
delay=net.delay; %Verzögerungen

%Definitionen
X=[];   % Menge der Eingangsschichten( Eingangsgewichte oder Verzögerungen >0 bei Verbindungsgewichten)
U=[];   % Menge der Ausgangsschichten (Ausgang der Schihct geht in Berechnung der KF ein oder geh über Verzögerung > 0 in eine Eingangsschicht)
I=cell(M,1); %Menge der Eingänge, die in Schicht m einkoppeln



%---------------------------
% Eingänge koppeln nur in Schicht 1 ein
dI{1,1}=delay.In;   %Eingangsverzögerungen vonzu Schicht 1 von Eingang 1
for d=dI{1,1}   %all Eingangsverzögerungen d
    IW{1,1,d+1}= (-0.5 + 1.*rand(layers(1),inputs));    %Eingangs-Gewichtmatrix P->S1    IW{d} steht dabei für die Verzögerung [d-1], da Matlab keine Null als Index nimmt
end 
X=[1];  %erste Schicht ist Eingangsschicht, da die Systemeingaänge hier einkoppeln
I{1}=1; % Menge der Eingänge, die in Schicht 1 einkoppeln (Nur ein Eingang)

%---------------------------------------
%Verbindungdgewichtsmatrizen erstellen
for m=1:M %Alle Schichten m
    L_b{m}=[]; % Menge der Schichten , die eine direkte Rückwärtsverbindung zu Schicht m besitzen
    L_f{m}=[]; %Menge der Schichten, die eine direkte Vorwärtsverbindung zur Schicht m besitzen
    
    %Vorwärtsverbindungen
    if m>1  %
        l=m-1;
        dL{m,l}=0;    %keine Verzögerun in den Vorwärtsverbindungen
        LW{m,l,1}=(-0.5 + 1.*rand(layers(m),layers(l)));    %Gewichtmatrix Sm -> Sm+1 Vorwärtsverbindung nur zur nächsten Schicht und ohne Verzögerung. Zufällige Werte zwischen -0,5 und 0,5
        L_b{l}=m; % Schicht m besitzt eine direkte Rückwärtsverbindung zur Schicht l
        L_f{m}=[L_f{m},l]; %Schicht l besitzt eine direkte Vorwärtsverbindung zur Schicht m
    end
    
    %Rückwärtsverbindungen
    for l=m:M % Es gibt mögliche Rückwärtserbindungen zur Schicht m von allen Schichten >= m
        
        if (m==1)&&(l==M)   %Sonderfall: Verzögerung von Ausgang zur Schicht 1
            dL{m,l}=delay.Out; %Verzögerungen der Ausgangsschicht zur Eingangsschicht 1
        else
            dL{m,l}=delay.Intern; %Alle anderen Schcihten habe zu sich selbst und zu allen vorherigen Schichten die Verzögerungen aus delay.Intern
        end
        
        for d=dL{m,l} %Alle Verzögerungen von Schicht l zur Schicht m
            LW{m,l,d+1}=(-0.5 + 1.*rand(layers(m),layers(l)));    %Gewichtmatrix Sl -> Sm für für Verzögerung d erstellen. Zufällige Werte zw -0,5 und 0,5
            if (sum(l==L_f{m})==0) % Falls l noch nicht in L_f{m} vorhanden
                L_f{m}=[L_f{m},l];  %l zur Menge L_f{m} hinzufügen
            end
            if (l>=m)&&(d>0) % Falls LW{m,l,d+1} eine Verzögerte Rückkopplung
                if (sum(m==X)==0) %Und falls m noch nicht in X vorhanden
                    X=[X,m];    % m zur Menge der Eingangsschichten hinzufügen
                end
                if (sum(l==U)==0) % Und falls l noch nicht in U vorhanden 
                    U=[U,l];    % l zur Menge der Ausgangsschichten hinzufügen
                end
            end
        end
    end
    
    b{m}=(-0.5 + 1.*rand(layers(m),1)); % Biasvektor Schicht m erzeugen. Zufällige Werte zw. -0,5 und 0,5
end
            
if (sum(M==U)==0) %Falls M noch nicht in U vorhanden
    U=[U,M];    % letzte Schicht ist Ausgangsschicht, da Systemausgang
end       

for  u=U %Für alle Ausgangsschichten
    CX_LW{u}=[]; %Menge aller Eingangsschichten, die ein Signal von u bekommen
    for x=X %für alle Eingangsschichten
        if (size(intersect(u,L_f{x}))>0)&(sum(x==CX_LW{u})==0)&(any(dL{x,u}>0)) %Falls u in L_f{x} UND x noch nicht in CX_LW{u} UND die Verbindung Schicht u -> Schicht x eine Verzögerung >0 besitzt
            CX_LW{u}=[CX_LW{u},x];   %c zur Menge CX_LW{u} hinzufügen
        end
    end
end

for x=1:M %Für alle Schichten
    CU_LW{x}=[]; %Menge aller Ausgangsschichten mit einer Verbindung zu x
    for u=U %Für alle Ausgangsschichten
        if any(dL{x,u}>0)   %Falls die Verbindung Schicht u -> Schicht x eine VErzögerung >0 besitzt 
            CU_LW{x}=[CU_LW{x},u];   %u zur Menge CU_LW{x} hinzufügen
        end
    end
end

net.U=U; % Menge aller Ausgangsschichten
net.X=X; % Menge aller Eingangsschichten
net.dL=dL; % Verzögerungen zwischen den Schichten
net.dI=dI; % Verzögerungen der Eingänge
net.L_b=L_b; %Rückwärtsverbindungen
net.L_f=L_f;  % Vorwärtsverbindungen
net.I=I;      % Eingangsmatrizen (hier nur eine Eingangsmatrix der in Schicht 1 einkoppelt)
net.CU_LW=CU_LW;  %CX_LW{u} = %Menge aller Eingangsschichten, die ein Signal von u bekommen  
net.CX_LW=CX_LW; %CU_LW{x} = Menge aller Ausgangsschichten mit einer Verbindung zu x

net.w_0=Wb2w(net,IW,LW,b);  %Gesamtgewichtsvektor aus Eingangs- und Verbindungsmatrizen sowie Biasvektoren erstellen
                    
    

