%Levenberg Marquardt (LM) mit Jakobimatrixberechnung durch RTRL

function net=train_LM(P,Y,net,k_max,E_stop)

dampconst   =     10;   %constant to adapt damping factor of LM
dampfac    =     3;    %damping factor of LM

[data,net] = prepare_data(P,Y,net);

[J,E,e]=RTRL(net,data);  %Jakobimatrix, Kostenfunktion und Fehlervektor des Startvektors  berechnen

k=1; %Erster Iterationsschritt
Ek(k)=E; %Verlauf der Kostenfunktion
disp(['Iteration: ', num2str(k),'   Error: ', num2str(E),'   SkalFakt:', num2str(dampfac)])

while 1

    JJ=J'*J; %J(transp) mal J
    w=net.w;    %Aktuellen Gewichtsvektor speichern
    
    while 1 %Bis Optimierungsschritt erfolgreich:
        
%         G=inv(JJ+SkalFakt.*eye(size(JJ,1))); %Skalierte Inverse Hessematrix berechnen
         G=(JJ+dampfac.*eye(size(JJ,1)))\eye(size(JJ+dampfac.*eye(size(JJ,1))));
        
        g=J'*e;  %Gradient
        if isnan(G(1,1))
            w_delta=-1/1e10.*g;
        else
            w_delta=-G*g;  %Gewichtsänderung ermitteln: w_delta=-G*g
        end
        net.w=w+w_delta; %Gewichte anpassen
    
        [E2] = calc_error(net,data); %Kostenfunktion an neuem Gewichtsvektor berechnen

        %-----------Optimierungsschritt erfolgreich------------------------
        if E2<E     
            dampfac=dampfac/dampconst;    %Skalierungsfaktor anpassen
            break;                          %Weiter
            
        %---------%Optimierungsschritt NICHT erfolgreich--------------------   
        elseif E2>=E    
            dampfac=dampfac*dampconst;     %Skalierungsfaktor anpassen       
        end          
    end
    
    [J,E,e,a]=RTRL(net,data);  %Jakobimatrix, Kostenfunktion und Fehlervektor an neuem Gewichtsvektor berechnen berechnen
    
    
    k=k+1;  %Aktueller Iterationsschritt
    Ek(k)=E; %Verlauf der Kostenfunktion
    disp(['Iteration: ', num2str(k),'   Error: ', num2str(E),'   SkalFakt:', num2str(dampfac)])

    if (k>=k_max) || (E<=E_stop) % Abbruch wenn eines der Abbruchkriterien erfüllt 
           break
    end    
    
end

%Ausgabe
net.ErrorHistory=Ek;

