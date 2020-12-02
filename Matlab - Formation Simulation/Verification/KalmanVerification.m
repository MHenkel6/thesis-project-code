close all
clc;
seed = 6;
rng(seed)
%% Script to verify Kalman Filter
% Continuous state space system
a = -0.4;
b = 1;
c = 1;
d = 0;
q = 0.2;
r = 3;

A = [a];
B = [b];
C = [c];
D = [d];
Q = [q];
R = [r];
E = [1];
% Disretize System
dt = 1;
sys = ss(A,B,C,D);
dSys = c2d(sys,dt,'zoh');
Ad = dSys.A;
Bd = dSys.B;
Cd = dSys.C;
Dd = dSys.D;

% Solve Discrete Algebraic Ricatti Equation
P = idare(Ad',Cd',Q,R,0,1);

% Solve Analytically
astar = (-Cd^2);
bstar = (Ad^2*r  +Cd^2*q- r);
cstar = q*r;

p = max((-bstar-sqrt(bstar^2-4*astar*cstar))/(2*astar),(-bstar+sqrt(bstar^2-4*astar*cstar))/(2*astar));

% Compare solution difference
pdiff = p-P
K = Ad*Cd*p/(Cd^2*p+r);

%% Simulate Kalman Filter
simEnd = 10000;
xArray = zeros(simEnd+1,1);
measurementArray = zeros(simEnd+1,1);
kalmanArray = zeros(simEnd+1,1);
residualArray = zeros(simEnd+1,1);
x = 1;
xArray(1) = x+sqrt(Q)*randn(1);
measurementArray(1) = x + sqrt(R)*randn(1);
kalmanArray(1) = measurementArray(1);
residualArray(1) = 0;
kalmanEstimate = kalmanArray(1);
ErrCov = q;
kArray = 1:simEnd+1;

for k = kArray(2:end)
    %Propagate State
    controlState = sin(2*pi*k/8);
    x = Ad*x+Bd*controlState +sqrt(Q)*randn(1) ;
    xArray(k) = x;
    % Take Measurement
    measurement = Cd*x + sqrt(R)*randn(1);
    measurementArray(k) = measurement;
    %Update Kalman Filter
    
    estimatePriori = Ad * kalmanEstimate + Bd * controlState;
    % Propagate error covariance matrix
    ErrCov = Ad*ErrCov*Ad'+Q;

    % Update estimate using measurement
    kalmanResidual =  (measurement-Cd*estimatePriori -Dd* controlState);
    kalmanEstimate = estimatePriori + K * kalmanResidual;
    kalmanArray(k) = kalmanEstimate;
    residualArray(k) = kalmanResidual;
    % Update Error Covariance Matrix
    I = eye(size(A));
    ErrCov = (I -K * Cd)*ErrCov*(I -K * Cd)' + K*R*K';
end

mediumSize = 13;
largeSize = 13;
lwidth = 1.;
figure
plot(kArray(1:50),xArray(1:50),'linewidth',lwidth)
hold on
plot(kArray(1:50), kalmanArray(1:50),'linewidth',lwidth)
plot(kArray(1:50),measurementArray(1:50),'linewidth',lwidth)
ax = gca;
ax.FontSize = mediumSize; 
legend("State","Kalman Filter Output","Measurements")
xlabel("Discrete Time [s]",'FontSize',mediumSize)
ylabel("State Excitation [-]",'FontSize',mediumSize)
title("Simplified Discrete Kalman Filter Test Case",'FontSize',largeSize)
grid on 
grid minor
figure
plot(kArray(1:50),residualArray(1:50),'linewidth',lwidth)
hold on
ax = gca;
ax.FontSize = mediumSize; 
xlabel("Discrete Time [s]",'FontSize',mediumSize)
ylabel("State Excitation [-]",'FontSize',mediumSize)
title("Simplified Discrete Kalman Filter Test Case",'FontSize',largeSize)
grid on 
grid minor

figure
histogram(residualArray,'Normalization', 'pdf')
hold on
[m,s] = normfit(residualArray)
y = normpdf(sort(residualArray),m,s);
plot(sort(residualArray),y,'linewidth',3)
ax = gca;
ax.FontSize = mediumSize; 
grid on
grid minor
xlabel("Kalman Filter Residual [-]",'FontSize',mediumSize)
ylabel("Probability Density Function [-]",'FontSize',mediumSize)
legend("Residual Distribution","Gaussian Fit")
