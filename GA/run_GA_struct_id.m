function out = run_GA_struct_id(seed, popSize, maxGen)
% Optimización DIRECTA de g=[k1..k5,c1..c5] con GA multi-objetivo (gamultiobj)
%  - f1: error temporal ponderado en aceleración (MSE)
%  - f2: desalineación modal (picos espectrales)
%  - f3: regularización física (Aineq*g<=0, suavidad, anclaje log)
%
% Requisitos: datos en 'datosFiltradosSismoYOct8.mat' con variables:
%   a (NxM), ar (Nx5), ab (Nx1), t (Nx1). (af es opcional; no se usa aquí)
%
% Parámetros opcionales:
%   seed   : RNG seed (default 1)
%   popSize: tamaño de población (default 220)
%   maxGen : generaciones (default 160)
%
% Salida:
%   out struct con g_best, métricas, frente de Pareto, etc.

if nargin < 1, seed = 1; end
if nargin < 2, popSize = 220; end
if nargin < 3, maxGen  = 160; end
rng(seed);  ticTotal = tic;

%% ========================= CARGA DE DATOS ===============================
S  = load('datosFiltradosSismoYOct8.mat','a','ar','ab','t');
a  = S.a;
ar = S.ar;
ab = S.ab(:);
t  = S.t(:);

assert(isvector(ab) && isvector(t) && size(ar,2)==5, ...
    'Se requieren: a (NxM), ar (Nx5), ab (Nx1), t (Nx1).');
N  = numel(ab);
assert(size(ar,1)==N && size(a,1)==N, 'Dimensiones inconsistentes a/ar vs ab/t.');

%% ========================= MODELO FÍSICO ================================
M = diag([11.773 9.17 9.14 9.12 9.08]);   % masas 5 GDL

% Límites para k y c
lb = [7000 7000 7000 7000 7000   0   0   0   0   0]';
ub = [12000 12000 12000 12000 12000 100 100 100 100 100]';
g_seed = [8000 8000 8000 8000 8000  20 20 20 20 20]';

% Restricciones lineales Aineq*g <= bineq (monotonía aproximada en k)
Aineq = [ 0.6 -1   0   0   0   0 0 0 0 0;
         -1.4  1   0   0   0   0 0 0 0 0;
           0  0.6 -1   0   0   0 0 0 0 0;
           0 -1.4  1   0   0   0 0 0 0 0;
           0   0  0.6 -1   0   0 0 0 0 0;
           0   0 -1.4  1   0   0 0 0 0 0;
           0   0   0  0.6 -1   0 0 0 0 0;
           0   0   0 -1.4  1   0 0 0 0 0;
           0   0   0   0   0  0.6 -1   0   0   0;
           0   0   0   0   0 -1.4  1   0   0   0;
           0   0   0   0   0   0  0.6 -1   0   0;
           0   0   0   0   0   0 -1.4  1   0   0;
           0   0   0   0   0   0   0  0.6 -1   0;
           0   0   0   0   0   0   0 -1.4  1   0;
           0   0   0   0   0   0   0   0  0.6 -1;
           0   0   0   0   0   0   0   0 -1.4  1];
bineq = zeros(16,1);

%% ========================= PONDERACIONES Y FFT =========================
lambda_lin    = 1e6;   % penalidad restricción lineal
lambda_smooth = 1e2;   % suavidad entre pisos
lambda_log    = 5e-1;  % anclaje log a g_seed

KFFT   = 2048;
band   = 2:round(KFFT/3);
w_time = time_weights_from_ab(ab);   % peso temporal (0.3..1) ~ energía de ab

%% ========================= FITNESS MULTI-OBJETIVO ======================
fitMO = @(g) fitness_multi_g(g, ar, ab, t, M, ...
    g_seed, Aineq, bineq, ...
    w_time, KFFT, band, lambda_lin, lambda_smooth, lambda_log);

nvars = 10;

%% ========================= POOL PARALELO (opcional) ====================
try
    p = gcp('nocreate');
    if isempty(p) || ~isa(p,'parallel.ProcessPool')
        if ~isempty(p), delete(p); end
        parpool("Processes");
    end
catch
    p = gcp('nocreate');
    if isempty(p), parpool('local'); end
end

%% ========================= GA MULTI-OBJETIVO ===========================
% Inicialización cercana a g_seed + resto aleatorio
nSeeds   = round(0.30*popSize);
sigma_g  = 0.10; % 10% del rango
seedCloud = repmat(g_seed(:).', nSeeds, 1) + sigma_g*(ub-lb)'.*randn(nSeeds, nvars);
seedCloud = min(max(seedCloud, lb'), ub');

randRest  = lb' + (ub'-lb').*rand(popSize-nSeeds, nvars);
initPop   = [seedCloud; randRest];

opts = optimoptions('gamultiobj', ...
    'PopulationSize', popSize, ...
    'InitialPopulationMatrix', initPop, ...
    'SelectionFcn', {@selectiontournament, 6}, ...
    'CrossoverFcn', @crossoverintermediate, ...
    'CrossoverFraction', 0.70, ...
    'MutationFcn', {@mutationgaussian, 0.20, 0.95}, ...
    'ParetoFraction', 0.35, ...
    'FunctionTolerance', 1e-12, ...
    'MaxGenerations', maxGen, ...
    'MaxStallGenerations', round(0.6*maxGen), ...
    'UseParallel', true, ...
    'Display','iter');

fprintf('Optimizando con GA-MO (nvars=10, pop=%d, gens=%d)...\n', popSize, maxGen);
ticGA = tic;
[GPareto, Fpareto] = gamultiobj(fitMO, nvars, Aineq, bineq, [], [], lb, ub, [], opts);
tGA = toc(ticGA);
fprintf('GA finalizado en %.2f s (%.2f min)\n', tGA, tGA/60);

%% ========================= SELECCIÓN "KNEE" ============================
[idxKnee, Fknee_n, scales] = select_knee(Fpareto);
g_best = GPareto(idxKnee,:).';

%% ========================= SIMULACIÓN FINAL ============================
[~, ~, Arel] = simulate_building_no_mex_ACCEL(g_best, ab, t, M);

% Métricas por piso
n=5; MSE_a=zeros(1,n); RMSE_a=MSE_a; MAE_a=MSE_a; R2_a=MSE_a;
for i=1:n
    [MSE_a(i), RMSE_a(i)] = metrics(Arel(:,i), ar(:,i));
    MAE_a(i) = mean(abs(Arel(:,i)-ar(:,i)),'omitnan');
    R2_a(i)  = safe_r2(Arel(:,i), ar(:,i));
end

%% ========================= IMPRESIÓN Y GRÁFICAS ========================
k = g_best(1:5).'; c = g_best(6:10).';
disp('--- Coeficientes óptimos (K y C) ---');
disp(array2table([k c], 'VariableNames', ...
     {'k1','k2','k3','k4','k5','c1','c2','c3','c4','c5'}));

disp('--- Métricas por piso (aceleración) ---');
T = table((1:5).', MSE_a.', RMSE_a.', MAE_a.', R2_a.', ...
    'VariableNames', {'Piso','MSE','RMSE','MAE','R2'});
disp(T);

% Curvas por piso
for piso=1:5
    figure('Name',sprintf('Piso %d - aceleración',piso));
    plot(t, ar(:,piso), '.r', 'MarkerSize', 3); hold on;
    plot(t, Arel(:,piso), '--b', 'LineWidth', 1.2); grid on;
    xlabel('t [s]'); ylabel(sprintf('a_%d [m/s^2]',piso));
    legend('Medida','Modelo');
    title(sprintf('Piso %d: RMSE=%.3g  R^2=%.3f',piso,RMSE_a(piso),R2_a(piso)));
end

% Frente de Pareto
figure('Name','Frente de Pareto 3D (GA)');
plot3(Fpareto(:,1), Fpareto(:,2), Fpareto(:,3), '.','MarkerSize',10); grid on; hold on;
plot3(Fpareto(idxKnee,1),Fpareto(idxKnee,2),Fpareto(idxKnee,3),'rp','MarkerSize',14,'MarkerFaceColor','y');
xlabel('f_1: tiempo (acc)'); ylabel('f_2: modal (acc)'); zlabel('f_3: físico');
title('Frente de Pareto (knee marcado)'); view(135,20);

tTotal = toc(ticTotal);
fprintf('\nTiempo total: %.2f s (%.2f min)\n', tTotal, tTotal/60);

%% ========================= SALIDA ======================================
out = struct();
out.g_best   = g_best;
out.k        = k;
out.c        = c;
out.MSE_a    = MSE_a;
out.RMSE_a   = RMSE_a;
out.MAE_a    = MAE_a;
out.R2_a     = R2_a;
out.GPareto  = GPareto;
out.Fpareto  = Fpareto;
out.idxKnee  = idxKnee;
out.Fknee_n  = Fknee_n;
out.scales   = scales;
out.Arel     = Arel;
out.t        = t;
out.M        = M;
out.time_ga_s = tGA;
out.time_total_s = tTotal;

end

%% ========================= SUBFUNCIONES ================================
function F = fitness_multi_g(g, ar, ab, t, M, ...
    g_seed, Aineq, bineq, w_time, KFFT, band, ...
    lambda_lin, lambda_smooth, lambda_log)
% g: 10x1 (k1..k5,c1..c5)

g = g(:);
[~,~,Arel] = simulate_building_no_mex_ACCEL(g, ab, t, M);  % N×5

% f1: error temporal ponderado (MSE en aceleración, sumado en pisos)
n=5; f1=0; wt = w_time(:); wt = wt/mean(wt);
for i=1:n
    ea = Arel(:,i) - ar(:,i);
    f1 = f1 + mean((ea.^2).*wt);
end

% f2: desalineación modal (picos) usando pisos 1 y 2
dt = mean(diff(t)); fs = 1/dt;
Nsig = size(Arel,1);
win  = 0.5 - 0.5*cos(2*pi*(0:Nsig-1)'/(Nsig-1));
[Sa1,fgrid] = spec_fft(ar(:,1).*win,  KFFT, fs);
[Sy1,~]     = spec_fft(Arel(:,1).*win, KFFT, fs);
[Sa2,~]     = spec_fft(ar(:,2).*win,  KFFT, fs);
[Sy2,~]     = spec_fft(Arel(:,2).*win, KFFT, fs);

fa = top2peaks(fgrid, Sa1) + top2peaks(fgrid, Sa2);
fy = top2peaks(fgrid, Sy1) + top2peaks(fgrid, Sy2);
fa = fa/2; fy = fy/2;
f2 = sum((fa - fy).^2);

% f3: regularización física
viol = Aineq * g - bineq;
pen_lin = sum(max(viol,0).^2);
dk = diff(g(1:5)); dc = diff(g(6:10));
pen_smooth = sum(dk.^2) + sum(dc.^2);
gpos = max(g, 1e-6);
pen_log = sum( (log(gpos) - log(g_seed)).^2 );
f3 = lambda_lin*pen_lin + lambda_smooth*pen_smooth + lambda_log*pen_log;

F = [f1, f2, f3];
end

function [X,V,Arel] = simulate_building_no_mex_ACCEL(g, ab, t, M)
% Integra por Tustin y devuelve desplazamiento (X), velocidad (V) y
% ACELERACIÓN RELATIVA (Arel) por piso (5 GDL).
k1=g(1);k2=g(2);k3=g(3);k4=g(4);k5=g(5);
c1=g(6);c2=g(7);c3=g(8);c4=g(9);c5=g(10);
n=5; Mi = inv(M); l = ones(n,1);

C=[c1+c2,-c2,0,0,0; -c2,c2+c3,-c3,0,0; 0,-c3,c3+c4,-c4,0; 0,0,-c4,c4+c5,-c5; 0,0,0,-c5,c5];
K=[k1+k2,-k2,0,0,0; -k2,k2+k3,-k3,0,0; 0,-k3,k3+k4,-k4,0; 0,0,-k4,k4+k5,-k5; 0,0,0,-k5,k5];

A=[zeros(n) eye(n); -Mi*K -Mi*C];
B=[zeros(n,1); -l];

dt=t(2)-t(1); I=eye(2*n);
Ad=(I-0.5*dt*A)\(I+0.5*dt*A);
Bd=(I-0.5*dt*A)\(dt*B);

N=numel(t); x=zeros(2*n,1);
X=zeros(N,n); V=zeros(N,n); Arel=zeros(N,n);

for k=1:N
    X(k,:) = x(1:n).';
    V(k,:) = x(n+1:end).';
    Arel(k,:) = (-Mi*(C*x(n+1:end) + K*x(1:n)) - l*ab(k)).';
    x = Ad*x + Bd*ab(k);
end
end

function [Pxx, f] = spec_fft(x, KFFT, fs)
% Periodograma lado único
x = x(:);  N = numel(x);
X  = fft(x, KFFT);
Px2 = (abs(X).^2) / (N*fs + eps);
if mod(KFFT,2)==0, K2 = KFFT/2 + 1; else, K2 = (KFFT+1)/2; end
Pxx = Px2(1:K2);
if K2>2, Pxx(2:end-1) = 2*Pxx(2:end-1); end
f = (0:K2-1).' * (fs / KFFT);
end

function fx = top2peaks(fgrid, S)
% Devuelve las 2 frecuencias de pico más altas (fallback robusto)
[~, idx] = sort(S(:), 'descend');
idx = unique(idx,'stable');
if numel(idx) >= 2
    f = sort(fgrid(idx(1:2)));
elseif numel(idx) == 1
    f = [fgrid(idx(1)); fgrid(idx(1))];
else
    f = [fgrid(1); fgrid(1)];
end
fx = f(:).';
end

function wt = time_weights_from_ab(ab)
p = ab.^2; p = p - min(p); p = p / max(p + eps);
wt = 0.3 + 0.7*p;
end

function [MSE, RMSE] = metrics(yhat, y)
e = yhat(:) - y(:);
MSE  = mean(e.^2,'omitnan');
RMSE = sqrt(MSE);
end

function r2 = safe_r2(yhat, y)
yhat = yhat(:); y = y(:);
ok = isfinite(yhat) & isfinite(y);
yhat = yhat(ok); y = y(ok);
if numel(y) < 3 || all(abs(y - mean(y)) < 1e-12)
    r2 = NaN; return;
end
ss_res = sum((y - yhat).^2);
ss_tot = sum((y - mean(y)).^2) + eps;
r2 = 1 - ss_res/ss_tot;
end

function [idxKnee, Fknee_n, scales] = select_knee(F)
Fmin = min(F,[],1); Fmax = max(F,[],1);
scales = Fmax - Fmin + eps;
Fn = (F - Fmin)./scales;
d  = sqrt(sum(Fn.^2,2));
[~, idxKnee] = min(d);
Fknee_n = Fn(idxKnee,:);
end
