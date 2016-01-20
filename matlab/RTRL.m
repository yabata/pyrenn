% RTRL Algorithmus
% Berechnung der Jakobimatrix J der Fehlerfläche 


function [J,E,e,a] = RTRL(net,data) 

P=data.P;   %Eingangsdaten
Y=data.Y;   %Ausgangsdaten des Systems
a=data.a;   %Schichtausgänge
q0=data.q0; %Ab q0.tem Trainingsdatum Ausgänge berechnen


M = net.M;      %Anzahl der Schichten des NN
U = net.U;      % Vektor mit allen Ausgangsschichten
X = net.X;      % Vektor mit allen Eingangsschichten
layers=net.layers;    %Aufbau des Netzes
dI=net.dI;      %Verzögerung der Eingänge
dL=net.dL;      %Verzögerung zwischen den Schichten
L_f=net.L_f;    %Vorwärtsverbindungen der Schichten
L_b=net.L_b;    % Rückwärtsverbindungen
I=net.I;        %Eingänge in die Schichten
CU_LW=net.CU_LW;    %CU_LW{x} Menge aller Ausgangsschichten mit einer VErbindung zu x

% Gesamtgewichtvektor in Gewichtmatrizen/ Vektoren umwandeln
[IW,LW,b]=w2Wb(net);

%% 1. Vorwärtspropagieren:
[Y_NN,n,a] = NNOut_(P,net,IW,LW,b,a,q0);     %Ausgänge, Summenausgänge und Schichtausgänge des NN berechnen


%% 2. Kostenfunktion berechnen:
Y_delta=Y-Y_NN;   %Fehlermatrix: Fehler des NN-Ausgangs bezüglich des Systemausgangs für jeden Datenpunkt
e=reshape(Y_delta,[],1);    %Fehlervektor (untereinander) [y_delta1_ZP1;y_delta2_ZP1;...;y_delta1_ZPX;y_delta2_ZPX]
E=e'*e; %Kostenfunktion: Summierter Quadratischer Fehler

%% 3. Backpropagation RTRL

Q = size(P,2);     %Anzahl der Datenpunkte mit "alte Daten"
Q0 = Q-q0;  %Anzahl der Datenpunkte ohne "alten Daten"

%Vordefinitionen
dAu_db=cell(M,1);       %Ableitung da(u)/db nach Biasvektoren
dAu_dIW=cell(size(IW)); %Ableitung da(u)/dIW nach Eingangs-Gewichtmatrizen
dAu_dLW=cell(size(LW)); %Ableitung da(u)/dLW nach Verbindungd-Gewichtmatrizen
S=cell(Q,M);            %Sensitivitätsmatrizen
dA_dw=cell(Q,max(U));   %Ableitung dA(u)/dw nach Gesamtgewichtsvektor
Cs=cell(max(U),1);  % Menge (alle m) der existierenden Sensitivttätsmatrizen von Schicht U: Cs{U}=Alle m für die S{q,U,m} existiert
CsX=cell(max(U),1); % Menge (alle x) der existierenden Sensitivttätsmatrizen von Schicht U: CsX{U}=Alle x für die S{q,U,x} existiert
%Cs und CsX werden während der Berchnung der Sensitivitäten erzeugt

%Initialisierung
J=zeros(Q0*layers(end),net.N); %Jakobimatrix

for q=1:q0
     for u=U     %Für alle u € U
        dA_dw{q,u}=zeros(layers(u),net.N);   %Initialisierung
     end
end

%--------Beginn RTRL -------------------------------   
    for q=q0+1:Q % Alle Trainingsdaten von q0+1 bis Q

        U_=[];  %Menge Notwendig für die Berechnung der Sensitivitäten, wird während der Berschnung erzeugt
        for u=U     %Für alle u € U
            Cs{u}=[];       %Initialisierung
            CsX{u}=[];      %Initialisierung
            dA_dw{q,u}=0;   %Initialisierung
        end

%---------------Sensitivitätsmatrizen berechnen----------------------------

        for m=M:-1:1  %m dekrementieren in Backpropagation Reihenfolge

            for u=U_    %alle u € U_
                S{q,u,m}=0; %Sensitivitätsmatrix von Schicht u zu Schicht m
                for l=L_b{m} % Alle Schichten mit direkter Rückwärtsverbindg Lb(m) zur Schicht m
                    S{q,u,m}=S{q,u,m}+(S{q,u,l}*LW{l,m,1})*diag((1-((tanh(n{q,m})).^2)),0); %Sensitivitätsmatrix Rekursiv berechnen
                end
                if all(m~=Cs{u})  % Falls m noch nicht in Cs{u}
                    Cs{u}=[Cs{u},m];    % m zur Menge Cs(u) hinzufügen
                    if any(m==X)  % Falls m € X
                        CsX{u}=[CsX{u},m];  %m zur Menge Csx(u) hinzufügen
                    end
                end
            end
            if any(m==U)  % Falls m € U
                if m==M %Falls m=M (Ausgangsschicht M besitzt keine Transferfunktion: a{M}=n{M})
                    S{q,m,m}=diag(ones(layers(M),1),0);  %Sensitivitätsmatrix S(M,M) berechnen
                else
                    S{q,m,m}=diag((1-((tanh(n{q,m})).^2)),0); %Sensitivitätsmatrix S(m,m) berechnen
                end
                U_=[U_,m];  % m zur Menge U' hinzufügen
                Cs{m}=[Cs{m},m];  %m zur Menge Cs(m) hinzufügen
                if any(m==X) %Falls m € X
                    CsX{m}=[CsX{m},m];  %m zur Menge Csx(m) hinzufügen
                end
            end
        end
        
%-------------- Ableitungen Berechnen----------------------------------------            
        for u=sort(U)     %Alle u € U inkrementiert in Simulationsreihenfolge
          
                    %------------Statische Ableitungsberechnung----------------------- 
           dAe_dw=[]; %Explizite Ableitung Ausgänge Schicht u nach allen Gewichten
           for m=1:M  %Alle Schichten m             
                %------------------
                %Eingangsgewichte
                if m==1
                    for i=I{m}  %Alle Eingänge i die in Schicht m einkoppeln
                        for d=dI{m,i}   % alle Verzögerungen i->m
                            if (sum(size(S{q,u,m}))==0)||(d>=q) %Falls keine Sensitivität vorhanden ODER d>=q:
                                dAu_dIW{m,i,d+1}=kron(P(:,q)',zeros(layers(u),layers(m)));   %ABleitung gleich NULL
                            else
                                dAu_dIW{m,i,d+1}=kron(P(:,q-d)',S{q,u,m});   %Ableitung Ausgang u nach IW{m,i,d+1}
                            end
                            dAe_dw=[dAe_dw,dAu_dIW{m,i,d+1}]; % An Gesamtableitungsvektor da(u)/dw anhängen
                        end
                    end
                end
                %---------------------
                %Verbindungsgewichte
                for l=L_f{m}  %Alle Schichten l die eine direkte Vrwärtsverbindung zu Schicht m haben
                    for d=dL{m,l}    % alle Verzögerungen l->m
                        if (sum(size(S{q,u,m}))==0)||(d>=q) %Falls keine Sensitivität vorhanden ODER d>=q:
                            dAu_dLW{m,l,d+1}=kron(a{q,l}',zeros(layers(u),layers(m)));   %ABleitung gleich NULL
                        else
                            dAu_dLW{m,l,d+1}=kron(a{q-d,l}',S{q,u,m});   %Ableitung Ausgang u nach LW{m,l,d+1}
                        end
                        dAe_dw=[dAe_dw,dAu_dLW{m,l,d+1}]; % An Gesamtableitungsvektor da(u)/dw anhängen
                    end
                end
                %--------------
                %Biasgewichte
                if (sum(size(S{q,u,m}))==0) %Falls keine Sensitivität vorhanden
                    dAu_db{m}=zeros(layers(u),layers(m));%Ableitung Ausgang u nach b{m} = NULL
                else
                    dAu_db{m}=S{q,u,m};     %Ableitung Ausgang u nach b{m}
                end
                dAe_dw=[dAe_dw,dAu_db{m}]; % An Gesamtableitungsvektor da(u)/dw anhängen
           end %end m

         
    %-----------dynamische Ableitungsberechnung------------------------------

            dAd_dw=0; %Gesamtsumme über alle x
            for x=CsX{u} % Alle x in CsX(u)
                sum_u_=0;  %Summe über alle u_
                for u_=CU_LW{x} %alle u_ in CU_LW{x}
                    sum_d=0; %Summe über alle d
                    for d=dL{x,u_} %Alle verzögerungen Schichtu_ nach Schicht x
                        if ((q-d)>0)&&(d>0) %Verzögerung kann nicht größer als aktueles Datum sein, NUR Verzögerungen >0
                            sum_d=sum_d+LW{x,u_,d+1}*dA_dw{q-d,u_}; %Summe über alle d
                        end
                    end
                    sum_u_=sum_u_+sum_d; %Summe über alle u_
                end
                if abs(sum(sum(sum_u_)))>0 %Falls sum_u gültiger Wert
                    dAd_dw=dAd_dw+S{q,u,x}*sum_u_; %dynamische Ableitungsberechnung aufsummieren
                end
            end

      %--------statischer + dynamischer Anteil-----------------------  

            dA_dw{q,u}=dAe_dw+dAd_dw;   %Gesamte Ableitungsberechnung Ausgang u nach Geasamtgewichtsvektor w

        end %end u
    
    %------ Jakobimatrix belegen-----------------------------
        J(((q-q0)-1)*layers(end)+1:(q-q0)*layers(end),:)=-dA_dw{q,M};         %Jakobimatrix
             
    end  %end q 
end
