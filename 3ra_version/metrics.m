function [MSE,RMSE,MAE,R] = metrics(xhat,xr)
% Versión numérica (double) robusta: sin vpa/symbolic
    xhat = double(xhat(:));
    xr   = double(xr(:));

    n = numel(xr);
    if n==0 || numel(xhat)~=n || any(~isfinite(xhat)) || any(~isfinite(xr))
        % Penalización si hay problemas
        MSE = 1e12; RMSE = 1e6; MAE = 1e6; R = 0; 
        return;
    end

    e   = xhat - xr;
    MSE = mean(e.^2);
    RMSE = sqrt(MSE);
    MAE = mean(abs(e));

    y_mean = mean(xr);
    SS_res = sum(e.^2);
    SS_tot = sum((xr - y_mean).^2);
    if SS_tot > 0
        R = 1 - SS_res/SS_tot;  % R^2
    else
        R = 0;
    end

    % Último aseguramiento de finitud
    if ~isfinite(MSE), MSE = 1e12; end
    if ~isfinite(RMSE), RMSE = 1e6; end
    if ~isfinite(MAE), MAE = 1e6; end
    if ~isfinite(R),   R = 0;     end
end



% function [MSE,RMSE,MAE,R]=metrics(xhat,xr)
% n=length(xr);
% y_mean = mean(xr);    % xr es la salida real que debe seguir
%     E = 0;
%     E_1 = 0;        % MSE
%     E_2 = 0;        % MAE
% %    E_3 = 0;        % RMSE
%     E_4 = 0;        % R2
% %    E_5 = 0;        % WI
%     for cont=1:n
%         E = xhat(cont)-xr(cont);              % error en cada iteración, xhat es mi salida del observador 
%         E_1 = E_1 + (E)^2;                    % acumulacion de MSE
%         E_2 = E_2 + abs(E);                   % acumulacion de MAE
%         E_4 = E_4 + (xr(cont)-y_mean)^2;      % SStot
% %        E_5 = E_5 + ((yT(cont)-y_mean)+ff(cont)-y_mean)^2; % denominador Wi
%     end
% 
%     MSE = vpa(E_1 /(n),4);                   % MSE
%     RMSE = sqrt(MSE);                         % RMSE
%     R = 1 - (E_1/E_4);                        % R2
%     %WI = 1 - E_1/E_5                        % WI
%     MAE = E_2 / n ;                           % MAE
% 
% end