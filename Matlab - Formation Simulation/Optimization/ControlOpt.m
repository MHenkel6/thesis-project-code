clear all;
close all;
clc;
addpath('..')
addpath('../HelperFunctions')
addpath('../DisturbanceForces')
addpath('../Environment')
addpath('../FormationObjects')
addpath('../snopt-matlab-3.0.0/matlab')
addpath('../snopt-matlab-3.0.0/matlab/util')
addpath('../snopt-matlab-3.0.0/mex')
%% Setup
% Initial values
global printDV;
printDV = 0;
seed = 137;
rng(seed); 
Re = Constants.graviPar.Re/1000;
h = 400;
i = 60/360*2*pi;
seed = 137;
rng(seed); 
n = 6; 
center = [Re+h,0,0]*1000;
muEarth = 3.986E14;
T = 2*pi*sqrt(((Re+h)*1000)^3/muEarth);
dist = 100;
parameters = [];
nImpulse = 10;

t0 = 0;
dt = 20;
nOrbit = 1;
tE = round(nOrbit*T/dt)*dt;
it= 2;
steps = tE/dt;
printDV = 1;
c= cost([1,0,0,0,0.5,0.5,0.5])


%% Optimization
printDV = 0;
options = optimoptions('fmincon','Display','iter','MaxIterations',200);
x0 = [1,0,0,0,0.5,0.5,0.5];
x0 = [0.01,0.01,0.01];
X = 0:0.1:1;

A = [];
b = [];
Aeq = [];
beq = [];
lb= [0,0,0,0,0,0,0];
lb = [0,0,0];
ub = [1,1,1,1,1,1,1];
ub = [1,1,1];
nonlcon = @quatCon;
nonlcon = [];
%[x,minCost] = fmincon(@cost,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)
%% Brute forcing
step = 0.1;
ii = 1;
jj = 1; 
kk = 1;
C = zeros(1/step+1,1/step+1,1/step+1);
for dist = 0:step:1
    jj = 1;
    for h = 0:step:1
        kk = 1;
        for inc = 0:step:1
            C(ii,jj,kk) = cost([dist,h,inc]);
            kk = kk + 1;
        end
        jj = jj + 1;
    end
    ii = ii + 1 
end
%% Post Processing
for ii = 1:11
    avg1(ii) = mean(mean(C(ii,:,:)));
    avg2(ii) = mean(mean(C(:,ii,:)));
    avg3(ii) = mean(mean(C(:,:,ii)));
end
plot(avg1)
figure
plot(avg2)
figure
plot(avg3)
% 'State' for cost function and optimization should be the orbital
% parameters of the reference orbit and the orientation and offset of the
% formation orientation. 

% Define cost function
function C = cost(state)
% Cost is based on three factors: quality  of the formation, total DeltaV
% cost of keeping said formation and the spread of the DeltaV
    global printDV;
    distMin = 100;
    distMax = 1000;
    
    hMin = 350;
    hMax = 1000;
    
    incMin = 0;
    incMax = pi/2;
    
    qOrientation = [1,0,0,0];
    dist = distMin + (distMax-distMin)*state(1);
    h = hMin + (hMax-hMin)*state(2);
    inc = incMin + (incMax-incMin)*state(3);
    
    w1 = 1;
    w2 = 0;
   
   
    GM = Constants.muEarth;
    Re = 6371;
    n = 6; 
    type = 'octahedron';
    center = [Re+h,0,0]*1000;

    velCenter = [0,sqrt(GM/(norm(center)))*cos(inc),sqrt(GM/(norm(center)))*sin(inc)]; % Reference velocity of center point
    muEarth = GM;
    T = 2*pi*sqrt(((Re+h)*1000)^3/muEarth);

    formationOrientation = qOrientation;
    % Control Parameters
    controltype = 2;
    disctype = 2;
    thrustInterval = 30;
    burnTime = 10;

    % Spacecraft Parameters
    mass = 100; 
    spacecraftSize = [1,1,1];
    inertia = [100,0,0;
               0,100,0;
               0,0,100];

    isp = 200;
    mib = 0.00001;
    selfBiasSize = 0.5;
    relBiasSize = 0.05;
    velRelBiasSize = 0.0005;
    selfNoise = 1;
    relNoise = 0.02;
    velRelNoise = 0.0005;
    spacecraftParameters = {mass, spacecraftSize, inertia, ...
                  thrustInterval,burnTime,isp,mib...
                  selfBiasSize, relBiasSize, velRelBiasSize...
                  selfNoise, relNoise,velRelNoise};

    % Initialize formation
    sFormation = formation(n,type,center,velCenter,dist,formationOrientation,...
                           controltype,disctype,spacecraftParameters);
    sFormation.disturbancesOn = 0;
    % Orbit time and time step
    t0 = 0;
    dt = Constants.dt;
    nOrbit = 0.1;
    tE = round(nOrbit*T/dt)*dt;
    for t = t0:dt:tE-dt
    % solve dynamics equation using rk4
        sFormation.rk4Prop(t,dt)
    end
    dVArray = zeros(6,1);
    for ii = 1:6
        dVArray(ii) = sFormation.spacecraftArray(ii).accumDV;
    end

    % Calculate cost from DeltaV and positions
    Q= 0;
    C = w1*mean(dVArray)/nOrbit+w2*std(dVArray)/mean(dVArray);

end

function [c,ceq] = quatCon(state)
% Function calculating quaternion constraint
quat = state(1:4);
c = [];
ceq= sqrt(sum(quat.^2))-1;
end

