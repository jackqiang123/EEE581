% the main file: particle filter method
clear all;
clc
load ('mymeasurew.mat'); % this is the default measurement matrix in w1 w2
%load ('w.mat')
load ('myprocessv.mat'); % this is the default process matrix in v1 v2
%load ('v.mat')
% if you want to use your own noise, define v1, v2, w1, w2 here. Uncommnet load ('myprocessv.mat') if want to use only new w, and vice versa.



%% here is the defin of the constant system parameters
T = 200/0.05; %time slot
N = 5; % dim of vector
delta = 0.05;
Q = zeros(5,5);
Q(3,3)  = var(v1);
Q(4,4)  = var(v2);

Rmean = [mean(w1) ; mean(w2)];     % an estimate of the noise
Rvariance = [var(w1) 0;0 var(w2)]; % this can be used to generated the distribution of estimate noise of the pdf

beta0 = 0.59783; % constant parameters
H0 = 13.406;
Gm0 = 3.9860*10^5;
R0 = 6374;
xr = 6374;
yr = 0; % end of the parameters



%% we first simulate the system, obtaining the true state and estimation

x = zeros(N,T); % true state vector     -------------------!!!!!!!!---------------------------------x(k), true value, used in RMS
z = zeros(2,T); % true output vector

x0 = [6500.4;349.14;-1.8093;-6.7967;0.6932];

x(:,1) = x0; % intialized

for t = 2:T
    
    betak = beta0*exp(x(5,t-1));
    Rk = sqrt(x(1,t-1)^2+x(2,t-1)^2);
    Gk = -Gm0/Rk^3;
    Vk = sqrt(x(3,t-1)^2+x(4,t-1)^2);
    dk = -betak*exp((R0-Rk)/H0)*Vk;
    
    x(1,t) = x(1,t-1)+delta*(x(3,t-1));
    x(2,t) = x(2,t-1)+delta*(x(4,t-1));
    x(3,t) = x(3,t-1)+delta*(x(3,t-1)*dk+Gk*x(1,t-1)+v1(t));
    x(4,t) = x(4,t-1)+delta*(x(4,t-1)*dk+Gk*x(2,t-1)+v2(t));
    x(5,t) = x(5,t-1);
    
    z(1,t) = sqrt((x(1,t)-xr)^2+(x(2,t)-yr)^2)+w1(t);
    z(2,t) = atan((x(2,t)-yr)/(x(1,t)-xr))+w2(t);
end

% end of the simulated system



%% we use particle filter method
%% filter building step

times = 5; %%% run 5 times, choose the smallest one%%%
rmsPF_bestsofar = 1;
xestimation = zeros(5,4000);
for kkk = 1 : times
    
    PFestX = zeros(5,T);  % estimation of par.filter result
    
    
    
    xfilterintial = x0;
    xfilterintial(5) = 0;
    PFestX(:,1) = xfilterintial;
    
    obs = @(x)[sqrt((x(1)-xr)^2+(x(2)-yr)^2);atan((x(2)-yr)/(x(1)-xr))] ; % this is the output function
    
    gen_sys_noise = @(u) mvnrnd(zeros(5,1), Q);         % sample from p_sys_noise (returns column vector) - > generate a noise
    p_obs_noise   = @(v) mvnpdf(v,Rmean,Rvariance); % we use guassion. but can be more careful design if not guaasion. for example, lapalace
    % we use lapalace transform  instead
    b1 = 1/4000*norm(w1'-mean(w1),1);
    b2 = 1/4000*norm(w1'-mean(w1),2);
    % we swtich between lapalce and mvnpdf
    if (var(w2) > 0.3)
        p_obs_noise   = @(v) (1/2/b1*exp(-1/b1*abs(v(1)-mean(w1))))*(1/2/b2*exp(-1/b2*abs(v(2)-mean(w2))));
    end
    gen_obs_noise = @(v) mvnrnd(Rmean,Rvariance);
    p_yk_given_xk = @(k, yk, xk) p_obs_noise(yk - obs(xk));
    
    
    
    gen_x0 = @(x) x0;
    
    pf.k               = 1;
    pf.Ns              = 500;
    pf.w               = zeros(pf.Ns, T);
    pf.particles       = zeros(5, pf.Ns, T);
    pf.gen_x0          = gen_x0;
    pf.p_yk_given_xk   = p_yk_given_xk;
    pf.gen_sys_noise   = gen_sys_noise;
    
    
    
    
    %% Estimate state by particle filter
    for t = 2:T
        obs = @(x)[sqrt((x(1)-xr)^2+(x(2)-yr)^2)+w1(t);atan((x(2)-yr)/(x(1)-xr))+w2(t)] ;
        p_yk_given_xk = @(k, yk, xk) p_obs_noise(yk - obs(xk));
        pf.gen_sys_noise   = gen_sys_noise;
        pf.k = t;
        betak = beta0*exp(x(5,t-1));
        Rk = sqrt(x(1,t-1)^2+x(2,t-1)^2);
        Gk = -Gm0/Rk^3;
        Vk = sqrt(x(3,t-1)^2+x(4,t-1)^2);
        dk = -betak*exp((R0-Rk)/H0)*Vk;
        
        sys = @(x,v)[x(1)+delta*x(3);x(2)+delta*x(4);x(3)+delta*(x(3)*dk+Gk*x(1)+v(3));x(4)+delta*(x(4)*dk+Gk*x(2)+v(4));x(5)];
        
        [PFestX(:,t), pf] = particle_filter(sys, z(:,t), pf);
        
        
        
        
    end
    
    rmsPF = 1/T*sqrt((PFestX(1,:)-x(1,:))*(PFestX(1,:)-x(1,:))'+(PFestX(2,:)-x(2,:))*(PFestX(2,:)-x(2,:))');% RMS computation
    
    if (rmsPF < rmsPF_bestsofar)
        rmsPF_bestsofar = rmsPF; %-------------------!!!!!!!!---------------------------------RMS, the best RMS out of 5 trials
        xestimation = PFestX;   % -------------------!!!!!!!!---------------------------------x(k|k), estiamte value (corresponding to the best), used in RMS
        
    end
    
   
    
    
end
 rmsPF_bestsofar