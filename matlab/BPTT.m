% BPTT Algorithmus
% Berechnung des Gradienten g der Fehlerfläche 

function [g,E] = BPTT(net,data) 

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
CX_LW=net.CX_LW;    %CX_LW{u}: Menge aller Eingangsschichten, die ein Signal von u bekommen

%% Gesamtgewichtvektor in Gewichtmatrizen/ Vektoren umwandeln
[IW,LW,b]=w2Wb(net);

%% 1. Vorwärtspropagieren:
[Y_NN,n,a] = NNOut_(P,net,IW,LW,b,a,q0);     %Ausgänge, Summenausgänge und Schichtausgänge des NN berechnen


%% 2. Kostenfunktion berechnen:
Y_delta=Y-Y_NN;   %Fehlermatrix: Fehler des NN-Ausgangs bezüglich des Systemausgangs für jeden Datenpunkt
e=reshape(Y_delta,[],1);    %Fehlervektor (untereinander) [y_delta1_ZP1;y_delta2_ZP1;...;y_delta1_ZPX;y_delta2_ZPX]
E=e'*e; %Kostenfunktion: Summierter Quadratischer Fehler

%% 3. Backpropagation Through Time

Q = size(P,2);     %Anzahl der Datenpunkte mit "alte Daten"
Q0 = Q-q0;  %Anzahl der Datenpunkte ohne "alten Daten"

%Vordefinitionen

S=cell(Q,M);                    %Sensitivitätsmatrizen
dE_dIW=cell(size(IW));          %Ableitungen dE/dIW nach Eingangs-Gewichtmatrizen
dE_dLW=cell(size(LW));          %Ableitungen dE/dLW nach Verbindungd-Gewichtmatrizen
dE_db=cell(M,1);                %Ableitung dE/db nach Biasvektoren
dEs_dAu=cell(M,1);              %Statische Ableitung dE/dA nach Ausgängen
dEd_dAu=cell(M,1);              %Dynamische Ableitung dE/dA nach Ausgängen
dE_dAu=cell(Q,M);               % Ableitung der Kostenfunktion nach den Ausgängen gesamt

%--------Beginn BPTT -------------------------------   
for q=Q:-1:q0+1
    U_=[];  %Menge Notwendig für die Berechnung der Sensitivitäten, wird während der Berschnung erzeugt
    for x=1:M
        CsU{x}=[]; %Initialisierung CsU{x}= Alle u € U, bei denen S{q,u,x} existiert. Diese MEnge wird während der Berechnung erzeugt
    end

%---------------Sensitivitätsmatrizen berechnen----------------------------

    for m=M:-1:1  %m dekrementieren in Backpropagation Reihenfolge
        for u=U_    %alle u € U_
            S{q,u,m}=0;  %Sensitivitätsmatrix von Schicht u zu Schicht m
            for l=L_b{m} % Alle Schichten mit direkter Rückwärtsverbindg Lb(m) zur Schicht m
                S{q,u,m}=S{q,u,m}+(S{q,u,l}*LW{l,m,1})*diag((1-((tanh(n{q,m})).^2)),0);  %Sensitivitätsmatrix Rekursiv berechnen
            end
            if all(u~=CsU{m})  % Falls u noch nicht in CsU{m}
                CsU{m}=[CsU{m},u];    % u zur Menge CsU(m) hinzufügen
            end 
        end
        if any(m==U)  % Falls m € U
            if m==M %Falls m=M (Ausgangsschicht M besitzt keine Transferfunktion: a{M}=n{M})
                S{q,m,m}=diag(ones(layers(M),1),0);  %Sensitivitätsmatrix S(M,M) berechnen
            else
                S{q,m,m}=diag((1-((tanh(n{q,m})).^2)),0); %Sensitivitätsmatrix S(m,m)
            end
            if all(m~=U_)  % Falls m noch nicht in U_
                U_=[U_,m];  % m zur Menge U' hinzufügen
            end
            if all(m~=CsU{m})  % Falls u noch nicht in CsU{m}
                CsU{m}=[CsU{m},m];    % m zur Menge CsU(m) hinzufügen
            end             
        end
    end
    %------------- Ableitungsberewchnung dE/dA nach den Schichtausgängen--------------------
    for u=sort(U,'descend') %alle u € U dekrementiert in Backpropagation Reihenfolge
        
        %------------Statisch----------------------- 
        if u==M    %Ausgangsschicht M
            dEs_dAu {u} = -2.*(Y_delta(:,q));
        else       % statische Ableitung Schicht für Schicht Rückwärts.                                
            dEs_dAu {u}= 0.*(LW{u+1,u,1}'*S{q,M,u+1}'*dEs_dAu {M});
        end                                                                               
    
        
        %-----------dynamisch-------------------
        dEd_dAu{u}=0;   %Initialisierung dynamische Ableitung E nach Ausgang u
        for x=CX_LW{u}   % Alle Schichten x aus C_X_LW(u) ( Alle Schichten, die ein Signal von A(u) bekommen)
            sum_d=0;    
            for d = dL{x,u} %Für alle VErzögerungen Schicht u -> Schicht x
                if (d<=(Q-q))%&&(d>0)        % Nur wenn aktuelles "Datum" größer als die Verzögerung
                    sum_u_=0;
                    for u_=CsU{x}   %Für alle u_ aus CsU{x} (S{q,u_,x} existiert)
                        sum_u_=sum_u_ + S{q+d,u_,x}'*dE_dAu{q+d,u_}; %Dynamische Ableitungsberechnung Teil 1 (Summe über alle u_)
                    end
                    sum_d=sum_d+ LW{x,u,d+1}'*sum_u_; %dynamische Ableitungsberechung Teil 2 (Summe über alle d)
                end
            end
            dEd_dAu{u}=dEd_dAu{u}+sum_d;  % Dynamische Ableitungsberechnung Teil 3 (Summe über alle x € CX_LW{u})
        end
        
        % Gesamte Ableitungsberechnung: Statischer + dynamischer Anteil
        dE_dAu{q,u}=dEs_dAu{u}+dEd_dAu{u};
        
    end %end u

    %------------------ Ableitungsberechnung dE/dw nach den Gewichten-----------------
    for m=1:M   %Schicht von Vorne nach hinten. Ableitungen nach  IW, LW und b berechnen und in Ableitungsvektor dE_dw umwandeln

        dm{m}=0;
        for u=CsU{m}  % Für alle u € CsU{m}
            dm{m}=dm{m} + S{q,u,m}'*dE_dAu{q,u}; %Hilfsableitung dm berechnen
        end

        %Eingangsgewichte ------------------------------
        if m==1
            for i=I{m}  %Alle Eingänge i die in Schicht m einkoppeln
                for d=dI{m,i}   % alle Verzögerungen i->m
                    if (q-d)>0  % Verzögerung muss kleiner als aktuelles Datum sein
                        dE_dIW{m,i,d+1} = dm{m} * P(:,q-d)';  %dE/dIW der ersten Schicht mit Eingang P
                    else
                        dE_dIW{m,i,d+1}=0.*IW{m,i,d+1}; % Sonst Ableitung=NULL
                    end
                end
            end
        end
        
        %Verbindungsgewichte ------------------------
        for l=L_f{m}  %Alle Schichten l die eine direkte Vrwärtsverbindung zu Schicht m ahben
            for d=dL{m,l}    % alle Verzögerungen l->m
                if (q-d)>0   % Verzögerung muss kleiner als aktuelles Datum sein
                    dE_dLW{m,l,d+1} = dm{m} * a{q-d,l}'; %dE/dLW der Verbindungsmatrizen berechnen
                else
                    dE_dLW{m,l,d+1}=0.*LW{m,l,d+1};   %Sonst Ableitung=NULL
                end

            end
        end
        
        %Biasgewichte---------------------------------
        dE_db{m} = dm{m}; %Ableitung nach den Biasgewichtsvektoren
        
        
    end %end m
    
    %Gesamtableitungsvektor (Gradient) für aktuelles Datum q
    dE_dw = Wb2w(net,dE_dIW,dE_dLW,dE_db); %Ableitungsmatrizen in Gesamtableitungsvektor umwandeln
        
    if q==Q %Falls Erster berechneter Datenpunkt
        g=dE_dw;         %Startwert der Summe
    else
        g=g+dE_dw;       % Gesamtgradientenvektor g(w)
    end
end %end q
end %end function
