function out = run_MOPSO_struct_id(seed, nPart, nIter)
% PSO MULTI-OBJETIVO (MOPSO) para identificar g=[k1..k5,c1..c5]
% Objetivos:
%   f1: error temporal ponderado (MSE en aceleración, suma de pisos)
%   f2: desalineación modal (picos espectrales)
%   f3: regularización física (Aineq*g<=0, suavidad, anclaje log)
%
% Requiere: 'datosFiltradosSismoYOct8.mat' con a (NxM), ar (Nx5), ab (Nx1), t (Nx1)
%
% Parámetros:
%   seed  : RNG (default 1)
%   nPart : # partículas (default 200)
%   nIter : # iteraciones (default 200)
%
% Salida: struct con g_best (knee), métricas, Pareto, etc.

if nargin < 1, seed  = 1;   end
if nargin < 2, nPart = 200; end
if nargin < 3, nIter = 200; end
rng(seed);
tAll = tic;

%% ========================= CARGA DE DATOS ===============================
S  = load('datosFiltradosSismoYOct8.mat','a','ar','ab','t');
a  = S.a;  ar = S.ar;  ab = S.ab(:);  t = S.t(:);
assert(isvector(ab) && isvector(t) && size(ar,2)==5, ...
    'Se requieren: a (NxM), ar (Nx5), ab (Nx1), t (Nx1).');
N  = numel(ab);
assert(size(ar,1)==N && size(a,1)==N, 'Dimensiones inconsistentes a/ar vs ab/t.');

%% ========================= MODELO FÍSICO ================================
M = diag([11.773 9.17 9.14 9.12 9.08]);   % masas 5 GDL

% Límites y semilla
lb = [7000 7000 7000 7000 7000   0   0   0   0   0]';
ub = [12000 12000 12000 12000 12000 100 100 100 100 100]';
g_seed = [8000 8000 8000 8000 8000  20 20 20 20 20]';

% Restricción lineal Aineq*g <= bineq (monotonía aprox en k)
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

%% ========================= PESOS/FFT ===================================
lambda_lin    = 1e6;
lambda_smooth = 1e2;
lambda_log    = 5e-1;
KFFT   = 2048;
band   = 2:round(KFFT/3);
w_time = time_weights_from_ab(ab);

% Fitness vectorial (3 objetivos)
fitMO = @(g) fitness_multi_g(g, ar, ab, t, M, ...
    g_seed, Aineq, bineq, w_time, KFFT, band, ...
    lambda_lin, lambda_smooth, lambda_log);

%% ========================= PARALELO ====================================
try
    p = gcp('nocreate');
    if isempty(p) || ~isa(p,'parallel.ProcessPool')
        if ~isempty(p), delete(p); end
        parpool("Processes");
    end
catch
    p = gcp('nocreate'); if isempty(p), parpool('local'); end
end

%% ========================= MOPSO BÁSICO ================================
nvars = 10;

% Hiperparámetros PSO
w_inertia = 0.72;
c1 = 1.49; c2 = 1.49;
vmax = 0.25*(ub-lb);
vmin = -vmax;

% Inicialización
X = lb' + (ub'-lb').*rand(nPart,nvars);
V = zeros(nPart,nvars);
Pbest = X;
F_X   = zeros(nPart,3);
F_P   = zeros(nPart,3);

% Eval inicial (parfor)
parfor i=1:nPart
    F_X(i,:) = fitMO(X(i,:)');
    F_P(i,:) = F_X(i,:);
end

% Archivo externo (no dominados)
ArchiveX = []; ArchiveF = [];
[ArchiveX, ArchiveF] = updateArchive(ArchiveX, ArchiveF, X, F_X);

fprintf('MOPSO: nPart=%d, nIter=%d | ND inicial=%d\n', nPart, nIter, size(ArchiveX,1));
ticPSO = tic;

for it = 1:nIter
    % Selección de líderes (por crowding)
    leadersIdx = selectLeaders(ArchiveF, nPart);

    % Actualización de velocidad/posición
    for i=1:nPart
        gBest = ArchiveX(leadersIdx(i),:); % líder para i
        r1 = rand(1,nvars); r2 = rand(1,nvars);
        V(i,:) = w_inertia*V(i,:) ...
               + c1*r1.*(Pbest(i,:) - X(i,:)) ...
               + c2*r2.*(gBest      - X(i,:));
        V(i,:) = max(min(V(i,:), vmax'), vmin');
        X(i,:) = X(i,:) + V(i,:);
        % reparar límites
        X(i,:) = min(max(X(i,:), lb'), ub');
        % pequeña proyección (clip suave) si viola Aineq
        viol = Aineq*X(i,:)' - bineq;
        if any(viol>0)
            X(i,:) = project_linear_ineq(X(i,:)', Aineq, bineq, lb, ub).';
        end
    end

    % Evaluación (parfor)
    parfor i=1:nPart
        F_X(i,:) = fitMO(X(i,:)');
    end

    % Actualización de pbest
    for i=1:nPart
        if dominates(F_X(i,:), F_P(i,:))
            Pbest(i,:) = X(i,:);
            F_P(i,:)   = F_X(i,:);
        elseif ~dominates(F_P(i,:), F_X(i,:)) % no dominados mutuamente: usar crowding
            % empuja a menor suma normalizada
            if sum(F_X(i,:)) < sum(F_P(i,:))
                Pbest(i,:) = X(i,:);
                F_P(i,:)   = F_X(i,:);
            end
        end
    end

    % Actualiza archivo externo
    [ArchiveX, ArchiveF] = updateArchive(ArchiveX, ArchiveF, X, F_X);

    if mod(it,10)==0 || it==1
        fprintf('it=%4d | ND=%d | f1~%.3g f2~%.3g f3~%.3g (mejor)\n', ...
            it, size(ArchiveX,1), min(ArchiveF(:,1)), min(ArchiveF(:,2)), min(ArchiveF(:,3)));
    end
end
tPSO = toc(ticPSO);
fprintf('PSO finalizado en %.2f s (%.2f min) | ND=%d\n', tPSO, tPSO/60, size(ArchiveX,1));

%% ========================= SELECCIÓN KNEE ================================
[idxKnee, Fknee_n, scales] = select_knee(ArchiveF);
g_best = ArchiveX(idxKnee,:).';

%% ========================= SIMULACIÓN Y MÉTRICAS ========================
[~, ~, Arel] = simulate_building_no_mex_ACCEL(g_best, ab, t, M);

n=5; MSE_a=zeros(1,n); RMSE_a=MSE_a; MAE_a=MSE_a; R2_a=MSE_a;
for i=1:n
    [MSE_a(i), RMSE_a(i)] = metrics(Arel(:,i), ar(:,i));
    MAE_a(i) = mean(abs(Arel(:,i)-ar(:,i)),'omitnan');
    R2_a(i)  = safe_r2(Arel(:,i), ar(:,i));
end

k = g_best(1:5).'; c = g_best(6:10).';
disp('--- Coeficientes óptimos (K y C; solución knee) ---');
disp(array2table([k c], 'VariableNames', ...
     {'k1','k2','k3','k4','k5','c1','c2','c3','c4','c5'}));
disp('--- Métricas por piso (aceleración) ---');
T = table((1:5).', MSE_a.', RMSE_a.', MAE_a.', R2_a.', ...
    'VariableNames', {'Piso','MSE','RMSE','MAE','R2'});
disp(T);

% Curvas por piso
for piso=1:5
    figure('Name',sprintf('Piso %d - aceleración (MOPSO)',piso));
    plot(t, ar(:,piso), '.r', 'MarkerSize', 3); hold on;
    plot(t, Arel(:,piso), '--b', 'LineWidth', 1.2); grid on;
    xlabel('t [s]'); ylabel(sprintf('a_%d [m/s^2]',piso));
    legend('Medida','Modelo');
    title(sprintf('Piso %d: RMSE=%.3g  R^2=%.3f',piso,RMSE_a(piso),R2_a(piso)));
end

% Frente Pareto
figure('Name','Frente de Pareto 3D (MOPSO)');
plot3(ArchiveF(:,1), ArchiveF(:,2), ArchiveF(:,3), '.','MarkerSize',10); grid on; hold on;
plot3(ArchiveF(idxKnee,1),ArchiveF(idxKnee,2),ArchiveF(idxKnee,3), ...
      'rp','MarkerSize',14,'MarkerFaceColor','y');
xlabel('f_1: tiempo (acc)'); ylabel('f_2: modal (acc)'); zlabel('f_3: físico');
title('Frente de Pareto (knee marcado)'); view(135,20);

tTotal = toc(tAll);
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
out.ParetoX  = ArchiveX;
out.ParetoF  = ArchiveF;
out.idxKnee  = idxKnee;
out.Fknee_n  = Fknee_n;
out.scales   = scales;
out.Arel     = Arel;
out.t        = t;
out.M        = M;
out.time_pso_s   = tPSO;
out.time_total_s = tTotal;

end

%% ========================= SUBFUNCIONES ================================
function [ArchiveX, ArchiveF] = updateArchive(ArchiveX, ArchiveF, X, F)
% Añade candidatos no dominados y filtra dominados del archivo externo
if isempty(ArchiveX)
    ND = paretoMask(F);
    ArchiveX = X(ND,:); ArchiveF = F(ND,:);
else
    Cx = [ArchiveX; X];
    Cf = [ArchiveF; F];
    ND = paretoMask(Cf);
    ArchiveX = Cx(ND,:); ArchiveF = Cf(ND,:);
end
% (opcional) truncar por crowding si se vuelve muy grande
maxArch = 400;
if size(ArchiveX,1) > maxArch
    cd = crowdingDistances(ArchiveF);
    [~,ord] = sort(cd,'descend');
    ArchiveX = ArchiveX(ord(1:maxArch),:);
    ArchiveF = ArchiveF(ord(1:maxArch),:);
end
end

function leadersIdx = selectLeaders(F, nPick)
% Selecciona líderes por ruleta según crowding distance
cd = crowdingDistances(F);
p  = cd / (sum(cd)+eps);
edges = min([0; cumsum(p)],1); edges(end)=1;
u = rand(nPick,1);
leadersIdx = arrayfun(@(x)find(edges>=x,1,'first'), u);
leadersIdx = max(min(leadersIdx, size(F,1)),1);
end

function mask = paretoMask(F)
% Máscara de no-dominados (minimización)
N = size(F,1);
mask = true(N,1);
for i=1:N
    if ~mask(i), continue; end
    di = F(i,:);
    dom = all(bsxfun(@le, F, di),2) & any(bsxfun(@lt, F, di),2);
    dom(i) = false;
    if any(dom), mask(i) = false; end
end
end

function tf = dominates(fa, fb)
tf = all(fa <= fb) && any(fa < fb);
end

function cd = crowdingDistances(F)
% Distancia de crowding (más grande == más aislado)
M = size(F,2); N = size(F,1);
cd = zeros(N,1);
for m=1:M
    [fm, idx] = sort(F(:,m));
    cd(idx(1))   = inf;
    cd(idx(end)) = inf;
    denom = fm(end)-fm(1) + eps;
    for k=2:N-1
        cd(idx(k)) = cd(idx(k)) + (fm(k+1)-fm(k-1))/denom;
    end
end
end

function xproj = project_linear_ineq(x, A, b, lb, ub)
% Proyección muy simple: pasos de gradiente sobre violación y clamp en caja
xproj = min(max(x, lb), ub);
viol = A*xproj - b;
if any(viol>0)
    g = A(viol>0,:)'*ones(sum(viol>0),1);
    eta = 1e-4;
    xproj = xproj - eta*g;
    xproj = min(max(xproj, lb), ub);
end
end

function F = fitness_multi_g(g, ar, ab, t, M, ...
    g_seed, Aineq, bineq, w_time, KFFT, band, ...
    lambda_lin, lambda_smooth, lambda_log)

g = g(:);
[~,~,Arel] = simulate_building_no_mex_ACCEL(g, ab, t, M);

% f1: error temporal ponderado
n=5; f1=0; wt = w_time(:); wt = wt/mean(wt);
for i=1:n
    ea = Arel(:,i) - ar(:,i);
    f1 = f1 + mean((ea.^2).*wt);
end

% f2: desalineación modal (pisos 1 y 2)
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

% f3: físico
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
% Integración Tustin
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
x = x(:);  N = numel(x);
X  = fft(x, KFFT);
Px2 = (abs(X).^2) / (N*fs + eps);
if mod(KFFT,2)==0, K2 = KFFT/2 + 1; else, K2 = (KFFT+1)/2; end
Pxx = Px2(1:K2);
if K2>2, Pxx(2:end-1) = 2*Pxx(2:end-1); end
f = (0:K2-1).' * (fs / KFFT);
end

function fx = top2peaks(fgrid, S)
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
