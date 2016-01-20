%Quasi Newton Verfahren (BFGS) mit Ableitungsberechnung BPTT

function net=train_BFGS(P,Y,net,k_max,E_stop)

[data,net] = prepare_data(P,Y,net);

G=eye(length(net.w));         %Anfangsinitialisierung: Positiv definite Hessematrix
[g,E] = BPTT(net,data);  %Gradient und Kostenfunktion von Startpunkt berechnen

k=1;   %Aktueller Iterationsschritt
Ek(1)=E; %Verlauf der Kostenfumktion
disp(['Iteration: ', num2str(k),'   Error: ', num2str(E)])

while 1
  
    d_k=-G*g;  %Suchrichtung berechnen
    d_k=d_k./ sqrt(sum(d_k.^2));  %Suchrichtung Normieren
    
    eta_k = ALIS(net,data,d_k,E,k,4);    %Schrittweite eta_k in aktuelle Suchrichtung mit Liniensuche bestimmen

    w_delta=eta_k.*d_k;                       % delta w_k berechnen   
    net.w=net.w+w_delta;                         % Optimierungsschritt ausführen: Neuer Gewichtsvektor w_k2

    [g2,E]= BPTT(net,data);            % Gradient und Kostenfunktion an der neuen Stelle berechnen
    g_delta=g2-g;                          % delta_g berechnen
    
    G= G+(1+(g_delta'*G*g_delta) / (w_delta'*g_delta))...     %Aproxximation der Inversen Hessematrix
        *((w_delta*w_delta') / (w_delta'*g_delta))...
        -(w_delta*g_delta'*G + G*g_delta*w_delta') / (w_delta'*g_delta);
    
    g=g2;                           %Neu berechnete Werte für nächsten Schritt übernehemen
  
    k=k+1;  %Iterationsschritt
    Ek(k)=E; %Verlauf der Kostenfunktion
    disp(['Iteration: ', num2str(k),'   Error: ', num2str(E)])
    
    if (k>k_max) || (E<=E_stop) % Abbruch falls eines der Abbruchkriterien erreicht
        break
    end

end
%Ausgabe
net.ErrorHistory=Ek;