% the main file: UKFmethod.m
clear all;
close all;
clc


load ('mymeasurew.mat'); % this is the default measurement matrix in w1 w2
%load ('w.mat')
load ('myprocessv.mat'); % this is the default process matrix in v1 v2
%load ('v.mat')

%% here is the defin of the constant system parameters
T = 200/0.05; %time slot
N = 5; % dim of vector
delta = 0.05;
%Q0 = 2.4064*10^-2; % process noise
Q = zeros(5,5);
Q(3,3)  = var(v1);
Q(4,4)  = var(v2);

Rmean = [mean(w1) ; mean(w2)]; % an estimate of the noise
Rvariance = [var(w1) 0;0 var(w2)];

beta0 = 0.59783; % constant parameters
H0 = 13.406;
Gm0 = 3.9860*10^5;
R0 = 6374;
xr = 6374;
yr = 0; % end of the parameters



%% we first simulate the system, obtaining the true state and estimation

x = zeros(N,T); % true state vector
z = zeros(2,T); % true output vector

x0 = [6500.4;349.14;-1.8093;-6.7967;0.6932]; % initial condition

x(:,1) = x0; % intialized

for t = 2:T
    betak = beta0*exp(x(5,t-1));% coffeicient of the equations
    Rk = sqrt(x(1,t-1)^2+x(2,t-1)^2);
    Gk = -Gm0/Rk^3;
    Vk = sqrt(x(3,t-1)^2+x(4,t-1)^2);
    dk = -betak*exp((R0-Rk)/H0)*Vk;
    % state update
    x(1,t) = x(1,t-1)+delta*(x(3,t-1));
    x(2,t) = x(2,t-1)+delta*(x(4,t-1));
    x(3,t) = x(3,t-1)+delta*(x(3,t-1)*dk+Gk*x(1,t-1)+v1(t));
    x(4,t) = x(4,t-1)+delta*(x(4,t-1)*dk+Gk*x(2,t-1)+v2(t));
    x(5,t) = x(5,t-1);
    %the output function
    z(1,t) = sqrt((x(1,t)-xr)^2+(x(2,t)-yr)^2)+w1(t);
    z(2,t) = atan((x(2,t)-yr)/(x(1,t)-xr))+w2(t);
end
% end of the simulated system





%% next we use the UKF method

UKFxestX = zeros(5,T);% estimation of UKF result
xfilterintial = x0;
xfilterintial(5) = 0;
UKFxestX(:,1) = xfilterintial; % intialized it by x0 guess.

%laststateUKF = x0; % s is for ukf initila state

P = eye(5)*10^(-6); % intial guess of state cov
P(5,5) = 1;


for t=2:T
    betak = beta0*exp(x(5,t-1));
    Rk = sqrt(x(1,t-1)^2+x(2,t-1)^2);
    Gk = -Gm0/Rk^3;
    Vk = sqrt(x(3,t-1)^2+x(4,t-1)^2);
    dk = -betak*exp((R0-Rk)/H0)*Vk;
    % the function used for UKF.
    sys = @(x)[x(1)+delta*x(3);x(2)+delta*x(4);x(3)+delta*(x(3)*dk+Gk*x(1));x(4)+delta*(x(4)*dk+Gk*x(2));x(5)];  % nonlinear state equations
    obs = @(x)[sqrt((x(1)-xr)^2+(x(2)-yr)^2);atan((x(2)-yr)/(x(1)-xr))] ; % this is the output function
    % the end of function defintion
    
    [UKFxestX(:,t) P] = ukf(sys,UKFxestX(:,t-1),P,obs,z(:,t),Q,Rvariance);
    % this is the UKF function, note that this function is total unknown of the system true state since there is no such input
    
end

rmsUKF = 1/T*sqrt((UKFxestX(1,:)-x(1,:))*(UKFxestX(1,:)-x(1,:))'+(UKFxestX(2,:)-x(2,:))*(UKFxestX(2,:)-x(2,:))')

plot(x(1,:),x(2,:),'b'); hold on;
plot(UKFxestX(1,:),UKFxestX(2,:),'r--'); hold on;
title('Tracking by UKF vs acutual trace', 'FontSize', 18);
xlabel('X1', 'FontSize', 18) % x-axis label
ylabel('X2', 'FontSize', 18) % y-axis label
grid on;



