%Liniensuche mit Lagrange Interpolation
%Finde eta_k für die Lagrange interpolation L vom grad r , dass E gleich Minimal


function [eta_k] = ALIS(net,data,d_k,E_k,k,r)
w_k=net.w; %Aktueller Gesamtgewichtsvektor

persistent eta_max; %Obergrenze Gültigkeitsbereich: Wert zwischen Funktionsaufrufen Speichern
if k==1
    eta_max = 0.5;  %Für ersten funktionsaufruf setzen 
end
eta_valid =1/4 *eta_max;    %Untergrenze Gültigkeitsbereich

E=zeros(1,r+1); %Definition E. Es werden r+1 Kostenfunktionswerte E benötigt
E(1)=E_k;       % E für aktuellen Gewichtvektor bereits bekannt. Ek Kostenfunktion für w=w_k)
eta_j=[0:r]./r.*eta_max;  %Stützstellen berechnen

for J=2:r+1;    %Kostenfunktion für die Stützstellen für Lagrange interpolation berechnen
    w=w_k+eta_j(J).*d_k;    % Gesamtgewichtsvektor= alter Gewichtsvektor+Suchrichtung*aktuelleStützstelle_j
    net.w=w;
    E(J)= calc_error(net,data);  %Kostenfunktion an der aktuellen Stützstelle
    
end

%Minimumsuche min E(eta_k) für E(w_k+eta_k*d_k)
options=optimset('TolX',eta_j(r+1).*1e-10); %Genauigkeitseinstellungen für fminbnd
eta_k = fminbnd(@ALIS_LI,eta_j(1),eta_j(r+1)+1e-2,options);  %Minimum eta_k der Hilfsfunktion ALIS_LI im aktuellen Bereich suchen;



%% Intervallgrenzen anpassen, bis gefundenes Minimum im Gültigkeitsbereich
while 1
    %-----------------------------------------
    if eta_k>=eta_max  %Intervallvergrößerung

        eta_max=2*eta_max;  %Intervallgrenze verdoppeln
        eta_valid=2*eta_valid;  %Gültigkeitsbereich verdoppeln

        eta_j(1:r/2+1)=([0:r/2]+2)./(2*r).*eta_max; %alte, bereits bekannte Werte übernehmen
        E(1:r/2+1)=E([1:r/2+1]+2);    %alte, bereits bekannte Werte übernehmen (r=4: E1=E3; E2=E4 ;E3=E5)

        eta_j(r/2+2:r+1)=([r/2+1:r])./r.*eta_max;   %neue Werte berechnen
        j_min_E=r/2+2;  %Grenzen für ALIS_LI bestimmen, dass bereits bekannte Werte nicht nochmal berechnet werden
        j_max_E=r+1;   
        for J=j_min_E:j_max_E;    %Kostenfunktion E nur berechnen, wo noch nicht bekannt
            w=w_k+eta_j(J).*d_k;    % Gesamtgewichtsvektor= alter Gewichtsvektor*Suchrichtung*aktuelleStützstelle_j
            net.w_k=w;
            E(J)= calc_error(net,data);  %Kostenfunktion an der aktuellen Stützstelle
        end

        options=optimset('TolX',eta_j(r+1).*1e-10); %Genauigkeitseinstellungen für fminbnd
        eta_k = fminbnd(@ALIS_LI,eta_j(1),eta_j(r+1)+1e-2,options);  %Neues Minimum eta_k der Hilfsfunktion ALIS_LI im aktuellen Bereich suchen;
        continue;
    end
    
    %-----------------------------------------------
    if eta_k < eta_valid  %Intervallverkleinerung

        eta_max=1/4*eta_max; %Intervallgrenze vierteln
        eta_valid=1/4*eta_valid; %Gültigkeitsbereich vierteln

       eta_j(r/2+2:r+1)=([r/2+1:r]-2).*eta_max; %alte Werte übernehmen (Index 1 bleibt gleich!,E1)
        E(r/2+2:r+1)=E((r/2+2:r+1)-2); %alte, bereits bekannte Werte übernehmen (r=4: E4=E2; E5=E3; E1 bleibt E1)

        eta_j(2:r/2+1)=(1:r/2)./r.*eta_max; %neue Werte berechnen
        j_min_E=2;  %Grenzen für ALIS_LI bestimmen, dass bereits bekannte Werte nicht nochmal berechnet werden
        j_max_E=r/2+1;    
        for J=j_min_E:j_max_E;    %Stützstellen für Lagrange interpolation
            w=w_k+eta_j(J).*d_k;    % Gesamtgewichtsvektor= alter Gewichtsvektor*Suchrichtung*aktuelleStützstelle_j
            net.w_k=w;
            E(J)= calc_error(net,data);  %Kostenfunktion an der aktuellen Stützstelle
        end

        options=optimset('TolX',eta_j(r+1).*1e-10); %Genauigkeitseinstellungen für fminbnd
        eta_k = fminbnd(@ALIS_LI,eta_j(1),eta_j(r+1)+1e-2,options);  %Neues Minimum eta_k der Hilfsfunktion ALIS_LI im aktuellen Bereich suchen;
        continue;
    end
    
    % eta_k liegt im Gültigkeitsbereich
    if (eta_k<=eta_max)&&(eta_k>=eta_valid)
        break; % Funktion beenden und eta_k ausgeben
    end
end


%% Hilfsfunktion ALIS_LI mit nur einem Parameter eta um Minimum  im angegebenen Bereich einfach berechnen zu können (nested function!)

function L=ALIS_LI(eta)  %Lagrangeinterpolation mit r+1 Stützstellen in Abhängigkeit von eta
    
    L_j=zeros(1,r+1); %Definition: "Vorterm" für Lagrange Interpolation jeder Stützstelle
    
    for j=1:r+1;    %Für alle Stützstellen der Lagrange interpolation
    L_h=zeros(1,r+1); %Produkt-Term der LAgrangeinterpolation für jede Stützstelle
        for h=1:r+1
            if h==j 
                 L_h(j)=1;
                 continue;
            end
            L_h(h)= (eta-eta_j(h))/(eta_j(j)-eta_j(h)); % Wird Benötigt zur Berechnung vom "Vorterm" für Lagrange Interpolation.
        end 
        L_j(j) = prod(L_h);     %"Vorterm" für Lagrange Interpolation an Stützstelle j   Lj(j)=Produkt(j=1:r+1)[L_h(j)]
    end
    L=sum(L_j.*E); %Lagrangeinterpolation: SUMME(j=1:r+1)[L_j(j)*E(j)]

end


end