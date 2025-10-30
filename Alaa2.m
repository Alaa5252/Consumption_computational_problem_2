function Alaa2
%  Computational Problem 2
%  Part 1 
% I include a small Howard policy-iteration step to speed convergence.
% I guard logs/roots with tiny eps values to avoid numeric crashes.
% I rely on Tauchen.m and cdf_normal.m.
rng(1);                           % reproducible sims
Tsim        = 1000;               % total simulation length
Burn        = 500;                % drop first 500
USE_EPS_AR1 = true;               % TRUE = draw normal eps and run AR(1)
                                   % FALSE = simulate the Tauchen Markov chain

%% Calibration
% discounting / interest rate
beta = 0.96;
r    = 0.04;

% wage AR(1):  w_{t+1} = (1 - rho)*wbar + rho*w_t + eps_{t+1},  eps ~ N(0,sigma^2)
rho   = 0.90;
sigma = 0.15;
wbar  = 2.50;

% discretizing the wage process (for Bellman and policy functions)
Ny   = 7;         % number of wage grid points (balance speed/accuracy)
mSD  = 3;         % +/- 3 sigmas in Tauchen
[wGrid, P] = Tauchen(Ny, wbar, rho, sigma, mSD);
% Note: I will still simulate "innovations" from Normal if USE_EPS_AR1=true.
% to feed those realizations into the discrete model, I map each realized
% w_t to its nearest grid point.

% preferences:  u(c, n) = log( c - Omega * n^{1+1/phi} / (1 + 1/phi) )
% intratemporal FOC: marginal disutility equals the wage inside the log.
% the log's 1/(c - D(n)) factor cancels, so:
%   D'(n) = Omega * n^(1/phi) = w  =>  n(w) = (w / Omega)^phi.
% this collapses the dynamic choice to a 1D problem in a' 
% Frisch elasticity and steady hours target (I normalize to 40/168 weekly)
phi  = 2.0;
n_ss = 40/168;

% calibrating Omega using the intratemporal condition at steady state:
%   wbar = Omega * (n_ss)^(1/phi)  ===>  Omega = wbar / (n_ss)^(1/phi).
Omega = wbar / (n_ss)^(1/phi);

% Novel natural borrowing limit with endogenous labor.
% exogenous income case: x = - y_min / r.
% Here income depends on labor, and labor depends on the wage. The safest
% assumption is to set the limit using (worst wage state, feasible labor at
% that wage). At w_min, the agent can still work n(w_min) = (w_min/Omega)^phi.
% The minimum sustainable flow is w_min * n(w_min). To ensure repayment is
% always possible without negative consumption, impose:
%   x = - [ w_min * n(w_min) ] / r.
w_min = wGrid(1);
n_min = max( (w_min / Omega)^phi , 1e-10 );
xNat  = - (w_min * n_min) / r;

% Asset grid:
Na   = 800;
aMax = 8 * (wGrid(end) * ( (wGrid(end)/Omega)^phi ));
A    = linspace(xNat, aMax, Na)';

% Precompute n(w) and disutility D(n(w)) once:
n_w   = max( (wGrid(:)/Omega).^phi , 1e-12 );                   
Dis_w = Omega .* ( n_w.^(1 + 1/phi) ) ./ (1 + 1/phi);          

% Tolerances and policy iteration:
tol          = 1e-9;
maxit        = 2e4;
howardIters  = 20;

% Initial value (zeros, I use Howard to accelerate)
V0 = zeros(Na, Ny);

% Solving: V(a,w), a'(a,w) and c(a,w)
[Va, polAprime_hh, ~] = vfi_household( ...
    A, wGrid, P, beta, r, n_w, Dis_w, V0, tol, maxit, howardIters);

% Ploting V(a,w) for every wage state:
figure('Name','Part I: Value Function','Color','w');
plot(A, Va, 'LineWidth', 1.0);
xlabel('Assets a'); ylabel('V(a,w)');
title('Household Value Function (all wage states)');
legend(compose('w=%.2f', wGrid), 'Location','southoutside','NumColumns',3);

% Simulating 1000 periods and dropping the first 500:
[w_idx, w_sim] = simulate_wage_path(USE_EPS_AR1, wGrid, P, wbar, rho, sigma, Tsim);

[~, n_path, c_path, ap_idx] = simulate_household( ...
    A, polAprime_hh, w_idx, wGrid, n_w, r, Tsim);

figure('Name','Part I: Simulation','Color','w');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
nexttile; plot(Burn+1:Tsim, w_sim(Burn+1:Tsim)); xlabel('t'); ylabel('w'); title('Wage');
nexttile; plot(Burn+1:Tsim, A(ap_idx(Burn+1:Tsim))); xlabel('t'); ylabel('a'''); title('Next-period Assets a''');
nexttile; plot(Burn+1:Tsim, n_path(Burn+1:Tsim)); xlabel('t'); ylabel('n'); title('Labor');
nexttile; plot(Burn+1:Tsim, c_path(Burn+1:Tsim)); xlabel('t'); ylabel('c'); title('Consumption');

% Standard deviation of n + qualitative expectations:
sd_n = std(n_path(Burn+1:Tsim));
fprintf('\n[Part I] Std dev of n (after burn-in %d): %.6f\n', Burn, sd_n);

fprintf('[Part I] Qualitative effects on sd(n):\n');
fprintf('  (a) Borrowing constraint = 0 (less tight)   -> LOWER sd(n)\n');
fprintf('  (b) Relative risk aversion doubled          -> LOWER sd(n)\n');
fprintf('  (c) Frisch elasticity doubled               -> HIGHER sd(n)\n');
fprintf('  (d) Real wage volatility doubled            -> HIGHER sd(n)\n');
fprintf('  (Note) We used the novel natural borrowing limit: x = -w_min*n(w_min)/r\n');

%% Part 2

% Technology:      f(uk,n) = (u k)^alpha * n^(1-alpha)
% Depreciation:    delta(u) = delta0 + phi1*(u-1) + (phi2/2)*(u-1)^2
% Law of motion:   k' = (1 - delta(u)) k + inv'
% Bellman with k' as the only dynamic choice (n and u are intratemporal):
%   V(k,w) = max_{u,n,k'} [ f(uk,n) - k' + (1 - delta(u))k - w n
%                            + beta E_w V(k',w') ]
%
% Intratemporal FOCs:
%   Labor:        f_n = (1-alpha)(uk)^alpha n^{-alpha} = w
%                 -> n*(k,u,w) = [((1-alpha)/w) (u k)^alpha]^{1/alpha}
%                               = ((1-alpha)/w)^{1/alpha} * u * k
%   Utilization:  alpha (u k)^{alpha-1} k n^{1-alpha} = delta'(u) k
%                 where delta'(u) = phi1 + phi2 (u - 1).
%   Plugging n(u) in, the LHS becomes constant in u for Cobb-Douglas,
%   so we obtain a simple solution:
%       u* = 1 + ( alpha * ((1-alpha)/w)^{(1-alpha)/alpha} - phi1 ) / phi2

alpha = 0.40;
delta0= 0.10;
phi2  = 0.20;
phi1  = 1/beta - (1 - delta0);   % ties discounting to depreciation cost

% Capital grid
Nk  = 600;
kMin= 1e-4;
kMax= 20;
K   = linspace(kMin, kMax, Nk)';

% Solving: V(k,w), k'(k,w), u(k,w), n(k,w), inv'(k,w)
[Vf, polKp, polU, polN, polInv] = vfi_firm( ...
    K, wGrid, P, beta, alpha, delta0, phi1, phi2, tol, maxit, howardIters);

% Plotting V(k,w) for all w:
figure('Name','Part II: Value Function','Color','w');
plot(K, Vf, 'LineWidth', 1.0);
xlabel('Capital k'); ylabel('V(k,w)');
title('Firm Value Function (all wage states)');
legend(compose('w=%.2f', wGrid), 'Location','southoutside','NumColumns',3);

% Simulating 1000 periods, dropping 500; show (w, k'', n, inv'', u) 
% Reusing the same wage path from Part 1 to focus on model differences.
[kp_idx, ~, u_path, n2_path, inv_path] = simulate_firm( ...
    K, polKp, polU, polN, polInv, w_idx, Tsim);

figure('Name','Part II: Simulation','Color','w');
tiledlayout(3,2,'Padding','compact','TileSpacing','compact');
nexttile; plot(Burn+1:Tsim, w_sim(Burn+1:Tsim)); xlabel('t'); ylabel('w'); title('Wage');
nexttile; plot(Burn+1:Tsim, K(kp_idx(Burn+1:Tsim))); xlabel('t'); ylabel('k'''); title('Next-period Capital k''');
nexttile; plot(Burn+1:Tsim, n2_path(Burn+1:Tsim)); xlabel('t'); ylabel('n'); title('Labor');
nexttile; plot(Burn+1:Tsim, inv_path(Burn+1:Tsim)); xlabel('t'); ylabel('inv'''); title('Investment');
nexttile; plot(Burn+1:Tsim, u_path(Burn+1:Tsim)); xlabel('t'); ylabel('u'); title('Utilization');

% Standard deviations of n, u, inv' + qualitative expectations:
sd_n2  = std(n2_path(Burn+1:Tsim));
sd_u   = std(u_path(Burn+1:Tsim));
sd_inv = std(inv_path(Burn+1:Tsim));
fprintf('\n[Part II] Std devs after burn-in %d:  sd(n)=%.6f,  sd(u)=%.6f,  sd(inv'''')=%.6f\n', ...
        Burn, sd_n2, sd_u, sd_inv);

fprintf('[Part II] Qualitative effects:\n');
fprintf('  (a) phi2 doubled  -> cost of deviating u from 1 is steeper:\n');
fprintf('      ==> LOWER sd(u), LOWER sd(inv''), LOWER sd(n)\n');
fprintf('  (b) real interest rate doubled (more impatience for today):\n');
fprintf('      ==> HIGHER sd(u), likely HIGHER sd(inv''), HIGHER sd(n)\n');

end

function [V, polAprime, polC] = vfi_household(A, wGrid, P, beta, r, n_w, Dis_w, V0, tol, maxit, howardIters)
% Solving the Part-1 household problem with endogenous labor handled
% analytically via n(w) = (w/Omega)^phi. Choice is only over a'
Na = numel(A); Ny = numel(wGrid);
V  = V0;                         % my initial guess
polAprime = ones(Na, Ny);        % policy indices 

onePlusR_A = (1+r)*A;            
Aprime     = A;                  

for it = 1:maxit
    % Expected value at each (a', current w): EV(a',w) = sum_{w'} V(a',w') P(w,w')
    EV = V * P';                 % Na x Ny

    % Howard improvement: evaluating current policy a' for a few steps
    for h = 1:howardIters
        for iy = 1:Ny
            ap_idx = polAprime(:,iy);                
            cnet   = onePlusR_A + (1+r)*(wGrid(iy)*n_w(iy)) - A(ap_idx);   % c before disutility
            util   = log( max(cnet - Dis_w(iy), 1e-14) );
            cont   = beta * EV(sub2ind([Na,Ny], ap_idx, iy*ones(Na,1)));
            V(:,iy)= util + cont;
        end
    end

    % Full maximization over all discrete a'
    V_new   = V;
    pol_new = polAprime;

    for iy = 1:Ny
        cnet = onePlusR_A + (1+r)*(wGrid(iy)*n_w(iy)) - Aprime';  
        uMat = log( max(cnet - Dis_w(iy), 1e-14) );
        val  = uMat + beta * (EV(:,iy))';                         
        [V_new(:,iy), ap_idx] = max(val, [], 2);
        pol_new(:,iy) = ap_idx;
    end

    if max(abs(V_new - V), [], 'all') < tol
        V = V_new; polAprime = pol_new;   
        break
    end
    V = V_new; polAprime = pol_new;
end

% Recovering consumption policy: c = (1+r)(a + w*n(w)) - a'
polC = zeros(Na,Ny);
for iy = 1:Ny
    ap = A(polAprime(:,iy));
    polC(:,iy) = (1+r)*A + (1+r)*(wGrid(iy)*n_w(iy)) - ap;
end
end


function [w_idx, w_sim] = simulate_wage_path(use_eps, wGrid, P, wbar, rho, sigma, T)
% Two options here:
%  Use Normal innovations and build the AR(1) directly (as the HW words it).
%  Or simulate directly from the Tauchen Markov chain.
Ny = numel(wGrid);

if use_eps
    eps = sigma * randn(T,1);
    w_sim = zeros(T,1);
    w_sim(1) = wbar;                        
    for t=2:T
        w_sim(t) = (1-rho)*wbar + rho*w_sim(t-1) + eps(t);
    end
    % Map continuous w to nearest discrete grid index
    w_idx = arrayfun(@(x) nearest_idx(wGrid, x), w_sim);
else
    % Markov chain simulation on Tauchen grid
    w_idx = zeros(T,1);
    w_idx(1) = ceil(Ny/2);                  
    for t=2:T
        u = rand;
        c = cumsum(P(w_idx(t-1),:));
        w_idx(t) = find(u <= c, 1, 'first');
    end
    w_sim = wGrid(w_idx);
end
end


function [a_path, n_path, c_path, ap_idx] = simulate_household(A, polAprime, w_idx, wGrid, n_w, r, T)
% Given policies and a wage-state path, simulate a, n, c.
Na     = numel(A);
a_idx  = ones(T,1) * ceil(Na/3);   % starting from a modest point on the grid
ap_idx = zeros(T,1);
a_path = zeros(T,1);
n_path = zeros(T,1);
c_path = zeros(T,1);

for t = 1:T
    a   = A(a_idx(t));
    iy  = w_idx(t);
    ap  = polAprime(a_idx(t), iy);
    ap_idx(t) = ap;

    n   = n_w(iy);                                % n(w) from intratemporal FOC
    c   = (1+r)*(a + wGrid(iy)*n) - A(ap);        % consumption

    a_path(t) = a;
    n_path(t) = n;
    c_path(t) = c;

    if t < T
        a_idx(t+1) = ap;
    end
end
end


function [V, polKp, polU, polN, polInv] = vfi_firm( ...
    K, wGrid, P, beta, alpha, delta0, phi1, phi2, tol, maxit, howardIters)
% Solving Part 2 firm problem. Intratemporal choices (u,n) are handled
% first; dynamic choice is over k' only. Howard steps speed convergence.

Nk = numel(K); Ny = numel(wGrid);
V  = zeros(Nk,Ny);
polKp = ones(Nk,Ny);
polU  = ones(Nk,Ny);
polN  = ones(Nk,Ny);
polInv= zeros(Nk,Ny);

% A modest feasible band for utilization (purely numerical safeguard)
uLo = 0.50; uHi = 1.80;

for it = 1:maxit
    EV = V * P';      

    % Howard improvement on current k' policy:
    for h = 1:howardIters
        for iy = 1:Ny
            w   = wGrid(iy);
            kp  = K(polKp(:,iy));   % chosen k'
            k   = K;

            [u_star, n_star] = solve_un(k, w, alpha, phi1, phi2, uLo, uHi);
            delta = delta0 + phi1*(u_star-1) + 0.5*phi2*(u_star-1).^2;
            f     = (u_star.*k).^alpha .* (n_star).^(1-alpha);
            flow  = f - kp + (1 - delta).*k - w.*n_star;

            V(:,iy)    = flow + beta * EV(sub2ind([Nk,Ny], polKp(:,iy), iy*ones(Nk,1)));
            polU(:,iy) = u_star;
            polN(:,iy) = n_star;
            polInv(:,iy) = kp - (1 - delta).*k;
        end
    end

    % Full maximization over discrete k':
    V_new   = V;
    pol_new = polKp;

    for iy = 1:Ny
        w = wGrid(iy);
        % Given (k,w), I compute u*(k,w) and n*(k,w) in closed form:
        [u_star, n_star] = solve_un(K, w, alpha, phi1, phi2, uLo, uHi);
        delta = delta0 + phi1*(u_star-1) + 0.5*phi2*(u_star-1).^2;
        f     = (u_star.*K).^alpha .* (n_star).^(1-alpha);
        curR  = f + (1 - delta).*K - w.*n_star;          % Nk x 1

        % Value for each candidate k':  curR - k' + beta * EV(k', current w)
        valMat = curR + beta * (EV(:,iy))';              % Nk x Nk
        valMat = valMat - K';                            % subtracting k' row-wise

        [V_new(:,iy), kp_idx] = max(valMat, [], 2);
        pol_new(:,iy) = kp_idx;

        polU(:,iy)   = u_star;
        polN(:,iy)   = n_star;
        polInv(:,iy) = K(kp_idx) - (1 - delta).*K;
    end

    if max(abs(V_new - V), [], 'all') < tol
        V = V_new; polKp = pol_new;   % converged
        break
    end
    V = V_new; polKp = pol_new;
end
end


function [kp_idx, k_path, u_path, n_path, inv_path] = simulate_firm( ...
    K, polKp, polU, polN, polInv, w_idx, T)
% Simulate the firm given discrete wage-state path and policies.
Nk = numel(K);
k_idx   = ones(T,1) * ceil(Nk/3);
kp_idx  = zeros(T,1);
k_path  = zeros(T,1);
u_path  = zeros(T,1);
n_path  = zeros(T,1);
inv_path= zeros(T,1);

for t = 1:T
    ik = k_idx(t);
    iw = w_idx(t);
    kp_idx(t) = polKp(ik, iw);

    k_path(t)   = K(ik);
    u_path(t)   = polU(ik, iw);
    n_path(t)   = polN(ik, iw);
    inv_path(t) = polInv(ik, iw);

    if t < T
        k_idx(t+1) = kp_idx(t);
    end
end
end


function [u_star, n_star] = solve_un(k, w, alpha, phi1, phi2, uLo, uHi)
% intratemporal solution for utilization and labor (vectorized in k).
% labor: n = ((1-alpha)/w)^{1/alpha} * u * k
coefN = ((1 - alpha)/w)^(1/alpha);

% Utilization FOC simplifies to an expression in u:
% alpha * coefN^(1-alpha)  =  phi1 + phi2 * (u - 1)
LHS = alpha * coefN^(1 - alpha);
if abs(phi2) > 1e-12
    u_star = 1 + (LHS - phi1)/phi2;
else
    u_star = 1;  % degenerate linear-cost case
end

% numerical stability
u_star = min(max(u_star, uLo), uHi);

% Labor from the closed form
n_star = coefN .* (u_star .* k);
n_star = max(n_star, 1e-12);     % guard
end


function idx = nearest_idx(grid, x)
% returns the index of the closest point in 'grid' to scalar x.
[~, idx] = min(abs(grid - x));
end
