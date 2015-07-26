function [xhk, pf] = particle_filter(sys, yk, pf)

k = pf.k;


%% Initialize variables
Ns = pf.Ns;                              % number of particles
nx = size(pf.particles,1);               % number of states

wkm1 = pf.w(:, k-1);                     % weights of last iteration
if k == 2
    for i = 1:Ns                          % simulate initial particles
        pf.particles(:,i,1) = pf.gen_x0(); % at time k=1
    end
    wkm1 = repmat(1/Ns, Ns, 1);           % all particles have the same weight
end



%% Separate memory
xkm1 = pf.particles(:,:,k-1); % extract particles from last iteration;
xk   = zeros(size(xkm1));     % = zeros(nx,Ns);
wk   = zeros(size(wkm1));     % = zeros(Ns,1);

%% Algorithm 3 of Ref [1]
for i = 1:Ns
    
    temp = pf.gen_sys_noise();
    xk(:,i) = sys(xkm1(:,i), temp);
    
    
    wk(i) = wkm1(i) * pf.p_yk_given_xk(k, yk, xk(:,i));
    if (wk(i) == 0)
        wk(i) = 0.001;
    end
    
end;

%% Normalize weight vector
wk = wk./sum(wk);

%% Calculate effective sample size: eq 48, Ref 1

[xk, wk] = resample(xk, wk);
% {xk, wk} is an approximate discrete representation of p(x_k | y_{1:k})

%% Compute estimated state
xhk = zeros(nx,1);
for i = 1:Ns;
    xhk = xhk + wk(i)*xk(:,i);
end

%% Store new weights and particles
pf.w(:,k) = wk;

pf.particles(:,:,k) = xk;

return; % bye, bye!!!

%% Resampling function
function [xk, wk, idx] = resample(xk, wk)

Ns = length(wk);  % Ns = number of particles



edges = min([0 cumsum(wk)'],1); % protect against accumulated round-off
edges(end) = 1;                 % get the upper edge exact
u1 = rand/Ns;
% this works like the inverse of the empirical distribution and returns
% the interval where the sample is to be found
[~, idx] = histc(u1:1/Ns:1, edges);

xk = xk(:,idx);                    % extract new particles
wk = repmat(1/Ns, 1, Ns);          % now all particles have the same weight

return; 
