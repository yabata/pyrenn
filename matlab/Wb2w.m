% Eingangs- und Verbindungsmatrizen sowie Biasvektoren für in Net
% vorgegebene  in Gesamtgewichtsvektor net.w_0 zusammenfassen

function w=Wb2w(net,IW,LW,b)

dL=net.dL; % Verzögerungen zwischen den Schichten
dI=net.dI; % Verzögerungen der Eingänge
I=net.I;      % Eingangsmatrizen (hier nur eine Eingangsmatrix der in Schicht 1 einkoppelt)
L_f=net.L_f;  % Vorwärtsverbindungen
M=net.M; % Anzahl der Schicjten des NN

w=[]; %Gesamtgewichtsvektor


for m=1:M  %Alle Schichten m
    
    %Eingangsgewichte
    if m==1 
        for i=I{m}  %Alle Eingangsmatrizen i die in Schicht m einkoppeln
            for d=dI{m,i}   % alle Verzögerungen i->m
                w=[w;IW{m,i,d+1}(:)];   %Eingangsgewichtsmatrix zu Gesamtgewichtsvektor hinzufügen [Matix(:) = vec(Matric)]
            end
        end
    end

    %Verbindungsgewichte
    for l=L_f{m}  %Alle Schichten l die eine direkte Vrwärtsverbindung zu Schicht m ahben
        for d=dL{m,l}    % alle Verzögerungen l->m
            w=[w;LW{m,l,d+1}(:)]; %Verbindungsgewichtsmatrix zu Gesamtgewichtsvektor hinzufügen  [Matix(:) = vec(Matric)]
        end
    end
    
    %Biasgewichte
    w=[w;b{m}];     %Biasvektor von Schicht m zum Gesamtgewichtsvektor hunzufügen
end