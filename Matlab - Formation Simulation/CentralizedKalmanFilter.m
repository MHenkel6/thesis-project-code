close all;
clear all;
clc;
addpath('HelperFunctions')
addpath('HelperFunctions/CW')
addpath('DisturbanceForces')
addpath('Environment')
addpath('Environment/igrf')
addpath('Environment/igrf/datfiles')
addpath('FormationObjects')
%% Setup
seed = 1371;
rng(seed); 
% Initial values
Re = Constants.graviPar.Re/1000;
h = 687.4800;
inc =  62.8110/360*2*pi;

nSat = 6; 
type = 'octahedron';
center = [Re+h,0,0]*1000;
GM = Constants.muEarth;
velCenter = [0,sqrt(GM/(norm(center)))*cos(inc),sqrt(GM/(norm(center)))*sin(inc)]; % Reference velocity of center point
muEarth = 3.986E14;
T = 2*pi*sqrt(((Re+h)*1000)^3/muEarth);
dist = 142.3900;% 0.0001;%
formationOrientation = [0.9328,-0.0854,0.0314,0.3488];
% Control Parameters
controltype = 2;
disctype = 2;
thrustInterval = 60;
burnTime = 30;
mass = 100; 
spacecraftSize = [1,1,1];
inertia = [100,0,0;
           0,100,0;
           0,0,100];
thrust = 4;
isp = 200;
mib = 0.005*0.1;

noiseFactor = 1;
selfBiasSize = noiseFactor*1;
relBiasSize = noiseFactor*0.05;
velRelBiasSize =  noiseFactor*0.0005;
selfNoise = noiseFactor*2;
relNoise = noiseFactor*0.02;
velRelNoise = noiseFactor*0.0005;

posErrWeight = 1e2; 
velErrWeight = 1e2;

navChange = 0;
if navChange
    nEnd = 5;
    paramArray = 0.5+0.5/nEnd*(1:nEnd);
    fileName = 'NavChange';
    rangeNoiseSize = 1.1e-4;
    rangeBiasSize = 1.1e-4;
    angleNoiseSize = 9.8e-5;
    angleBiasSize = 9.8e-5;
    navArray = [rangeNoiseSize,rangeBiasSize,angleNoiseSize,angleBiasSize];
else 
    navArray = 0;
end

spacecraftParameters = {mass, spacecraftSize, inertia, ...
              thrustInterval,burnTime,thrust,isp,mib...
              selfBiasSize, relBiasSize, velRelBiasSize...
              selfNoise, relNoise,velRelNoise,...
              posErrWeight,velErrWeight};
% Load residual vector
load('covariance_Mean_Empirical.mat','resMean')
% Initialize formation
sFormation = formation(nSat,type,center,velCenter,dist,formationOrientation,...
                       controltype,disctype,spacecraftParameters,0,resMean,...
                        navChange+1, navArray);
sFormation.disturbancesOn = 0;

faultType = 2;% faulttype, 1 is closed, 2 is open
faultTime = 1876;%1000 + rand() * T/2; % random time after settle down (1000s for Kalman filter) in the oscillation
faultParam = 0.1;
satNo = 1 ;
thrusterNo = 1;
sFormation.setFault(faultTime,satNo,thrusterNo,faultType,faultParam)

% Orbit time and time step
t0 = 0;
dt = Constants.dt;
nOrbit = 1;
tE = round(nOrbit*T/dt)*dt;
zeroed = false;

it= 1;
steps = round(tE/dt);
relMeasOutput = zeros((steps)*4*nSat,6);
extKalmanOutput = zeros(steps,72);
extResidualOutput = zeros(steps,144);
kalmanOutput = zeros(steps,72);
residualOutput = zeros(steps,144);
errOutput = zeros(steps*nSat,6);
commandOutput = zeros(steps*nSat,3);
posOutput = zeros((steps+1)*nSat,6);      
posOutput(1:nSat,1:6) = sFormation.getStates();

fdiOutput = zeros(steps,6*nSat);
detOutput = zeros(steps,1);
isoOutput = zeros(steps,1);

%% Integration
tic
for t = t0:dt:tE-dt
    % solve dynamics equation using rk4
    
    sFormation.rk4Prop(t,dt)
    [formationState,formationStateHill] =  sFormation.getRelStates(zeroed);
    relMeasOutput((it-1)*4*nSat+1:it*4*nSat,1:6) = formationState;
    absMeasOutput((it-1)*nSat+1:it*nSat,1:6) = sFormation.getAbsoluteMeasurement();
    relHillOutput((it-1)*4*nSat+1:it*4*nSat,1:6) = formationStateHill;
    [command, err] = sFormation.getControlCommands();
    commandOutput((it-1)*nSat+1 :it*nSat ,1:3)= command;
    errOutput((it-1)*nSat+1 :it*nSat ,1:6) = err;
   

    %extKalmanOutput(it,:) = sFormation.spacecraftArray(1).extKalmanEstimate;
    %extResidualOutput(it,:) = sFormation.spacecraftArray(1).extKalmanResidual;

    kalmanOutput(it,:) = sFormation.spacecraftArray(1).kalmanEstimate;
    residualOutput(it,:) = sFormation.spacecraftArray(1).kalmanResidual;
    
    fdiOutput(it,:) = sFormation.spacecraftArray(1).gk;
    detOutput(it,:) = sFormation.spacecraftArray(1).detect;
    isoOutput(it,:) = sFormation.spacecraftArray(1).isolate;
    it = it+1;
    posOutput((it-1)*nSat+1 :it*nSat ,1:6) = sFormation.getStates();
    % Check if orbits explode
    if any(euclidnorm(formationState(:,1:3)) - 1000*Re > 5E6 )
        break
    end
    
    percent = (it-2)/steps*100;% Percentage indicator, not simulation relevant
    if mod(percent,5) < 1/steps*100
        disp(string(round(percent))+ ' %');
    end
end
tElapsed = toc
%% Post Processing
mediumSize = 14;
posOutput = posOutput(1:end-6,:);
if faultType ~= 0 
    faultTime = sFormation.spacecraftArray(satNo).faultTime
else
    faultTime = 0.1;
end
faultVector = zeros(steps,1);
faultVector(ceil(faultTime):end) = (satNo-1)*6+thrusterNo;

% True positions
vecSat1 = posOutput(1:6:end,:);
vecSat2 = posOutput(2:6:end,:);
vecSat3 = posOutput(3:6:end,:);
vecSat4 = posOutput(4:6:end,:);
vecSat5 = posOutput(5:6:end,:);
vecSat6 = posOutput(6:6:end,:);

posSat1 = vecSat1(:,1:3);
posSat2 = vecSat2(:,1:3);
posSat3 = vecSat3(:,1:3);
posSat4 = vecSat4(:,1:3);
posSat5 = vecSat5(:,1:3);
posSat6 = vecSat6(:,1:3);
dist12 = euclidnorm(posSat1-posSat2);
dist13 = euclidnorm(posSat1-posSat3);
dist14 = euclidnorm(posSat1-posSat4);
dist15 = euclidnorm(posSat1-posSat5);


dist21 = dist12;
dist23 = euclidnorm(posSat2-posSat3);
dist25 = euclidnorm(posSat2-posSat5);
dist26 = euclidnorm(posSat2-posSat6);

dist31 = dist13;
dist32 = dist23;
dist34 = euclidnorm(posSat3-posSat4);
dist36 = euclidnorm(posSat3-posSat6);

dist41 = dist14;
dist43 = dist34;
dist45 = euclidnorm(posSat4-posSat5);
dist46 = euclidnorm(posSat4-posSat6);

dist51 = dist15;
dist52 = dist25;
dist54 = dist45;
dist56 = euclidnorm(posSat5-posSat6);

dist62 = dist26;
dist63 = dist36;
dist64 = dist46;
dist65 = dist56;


%%
% Measured positions
vecSatMeas1 = absMeasOutput(1:6:end,:);
vecSatMeas2 = absMeasOutput(2:6:end,:);
vecSatMeas3 = absMeasOutput(3:6:end,:);
vecSatMeas4 = absMeasOutput(4:6:end,:);
vecSatMeas5 = absMeasOutput(5:6:end,:);
vecSatMeas6 = absMeasOutput(6:6:end,:);
% Control commands
command1 = commandOutput(1:6:end,:);
command2 = commandOutput(2:6:end,:);
command3 = commandOutput(3:6:end,:);
command4 = commandOutput(4:6:end,:);
command5 = commandOutput(5:6:end,:);
command6 = commandOutput(6:6:end,:);
commandStack = [command1,command2,command3,command4,command5,command6];
%Errors
err1 = errOutput(1:6:end,:);
err2 = errOutput(2:6:end,:);
err3 = errOutput(3:6:end,:);
err4 = errOutput(4:6:end,:);
err5 = errOutput(5:6:end,:);
err6 = errOutput(6:6:end,:);
% determine relative distances
vecdist11 = vecSat2-vecSat1;
vecdist12 = vecSat3-vecSat1; 
vecdist13 = vecSat4-vecSat1;
vecdist14 = vecSat5-vecSat1;

vecdist21 = vecSat1-vecSat2;
vecdist22 = vecSat5-vecSat2; 
vecdist23 = vecSat6-vecSat2;
vecdist24 = vecSat3-vecSat2;

vecdist31 = vecSat1-vecSat3;
vecdist32 = vecSat2-vecSat3; 
vecdist33 = vecSat6-vecSat3;
vecdist34 = vecSat4-vecSat3;

vecdist41 = vecSat3-vecSat4;
vecdist42 = vecSat3-vecSat4; 
vecdist43 = vecSat5-vecSat4;
vecdist44 = vecSat1-vecSat4;

vecdist51 = vecSat4-vecSat5;
vecdist52 = vecSat6-vecSat5; 
vecdist53 = vecSat2-vecSat5;
vecdist54 = vecSat1-vecSat5;

vecdist61 = vecSat5-vecSat6;
vecdist62 = vecSat4-vecSat6; 
vecdist63 = vecSat3-vecSat6;
vecdist64 = vecSat2-vecSat6;
%}
%% Convert To Hill frame
%{
center = 1/6*(vecSat1+vecSat2+vecSat3+vecSat4+vecSat5+vecSat6);
vecdist12Hill = zeros(steps,3); 
vecdist13Hill = zeros(steps,3); 
vecdist14Hill = zeros(steps,3); 
vecdist15Hill = zeros(steps,3); 

vecdist23Hill = zeros(steps,3); 
vecdist34Hill = zeros(steps,3); 
vecdist45Hill = zeros(steps,3); 
vecdist52Hill = zeros(steps,3); 

vecdist65Hill = zeros(steps,3); 
vecdist64Hill = zeros(steps,3); 
vecdist63Hill = zeros(steps,3); 
vecdist62Hill = zeros(steps,3); 

vecSat1Hill = zeros(steps,3); 
vecSat2Hill = zeros(steps,3); 
vecSat3Hill = zeros(steps,3); 
vecSat4Hill = zeros(steps,3); 
vecSat5Hill = zeros(steps,3); 
vecSat6Hill = zeros(steps,3); 
a = sFormation.a;
n = sqrt(Constants.muEarth/a^3);
THillECIArray = zeros([3*steps,3]);
THillECIAnArray = zeros([3*steps,3]);
TdotArray =  zeros([3*steps,3]);
TtotalArray = zeros([6*steps,6]);
SArray =  zeros([3*steps,3]);
argLatArray = zeros([steps,1]);
nArray = zeros([steps,1]);

xArray = zeros([steps,3]);
yArray = zeros([steps,3]);
zArray = zeros([steps,3]);
x1Array = zeros([steps,3]);
y1Array = zeros([steps,3]);
z1Array = zeros([steps,3]);
z2Array = zeros([steps,3]);
for t = 1:steps
    posCenterECI = center(t,1:3);
    %  center velocity
    velCenterECI = center(t,4:6);
    sat1ECI = vecSat1(t,1:3);
    sat1vECI = vecSat1(t,4:6);
    [a,e,inc,O,~,theta,~,argLat,~] = rv2orb(posCenterECI',velCenterECI');
    [a,e,inc,O,~,theta,~,argLat,~] = rv2orb(sat1ECI',sat1vECI');

    n = sqrt(Constants.muEarth/(a^3*(1-e^2)^3))*(1+e*cos(argLat))^2;
    % Transform error into Hill Frame
    x = posCenterECI/norm(posCenterECI);
    z = cross(x,velCenterECI/norm(velCenterECI));
    y = cross(z,x)/(norm(cross(z,x)));
    x1 = sat1ECI/norm(sat1ECI);
    z1 = cross(x1,sat1vECI/norm(sat1vECI));
    y1 = cross(z1,x1)/(norm(cross(z1,x1)));
    THillECI = [x1;
                y1;
                z1];
    x2 = vecSat2(t,1:3)/norm(vecSat2(t,1:3));
    z2 = cross(x2, vecSat2(t,4:6)/norm(vecSat2(t,4:6)));
    S = n*skewSym(z);
    THillECIAn =rotZ(argLat)*rotX(inc)*rotZ(O);
    Tdot = n*rotZdot(argLat)*rotX(inc)*rotZ(O);
    Texp = rotZdot(argLat)*rotX(inc)*rotZ(O);
    Ttotal = [THillECI,zeros(3,3);
              -THillECI*Tdot*THillECI,THillECI];
    % determine relative distances
    vecdist12Hill(t,1:3)  = THillECI*vecdist11(t,1:3)'; 
    vecdist13Hill(t,1:3)  = THillECI*vecdist12(t,1:3)'; 
    vecdist14Hill(t,1:3)  = THillECI*vecdist13(t,1:3)'; 
    vecdist15Hill(t,1:3)  = THillECI*vecdist14(t,1:3)'; 
    
    vecdist23Hill(t,1:3)  = THillECI*vecdist24(t,1:3)'; 
    vecdist34Hill(t,1:3)  = THillECI*vecdist34(t,1:3)'; 
    vecdist45Hill(t,1:3)  = THillECI*vecdist43(t,1:3)'; 
    vecdist52Hill(t,1:3)  = THillECI*vecdist53(t,1:3)'; 
    
    vecdist65Hill(t,1:3)  = THillECI*vecdist61(t,1:3)'; 
    vecdist64Hill(t,1:3)  = THillECI*vecdist62(t,1:3)'; 
    vecdist63Hill(t,1:3)  = THillECI*vecdist63(t,1:3)'; 
    vecdist62Hill(t,1:3)  = THillECI*vecdist64(t,1:3)'; 
    
    % Determine Relative Velocities
    vecdist12Hill(t,4:6)  = THillECI*(vecSat2(t,4:6)-vecSat1(t,4:6))'-Tdot*THillECI*(vecSat2(t,1:3)-vecSat1(t,1:3))';
    vecdist13Hill(t,4:6)  = THillECI*(vecSat3(t,4:6)-vecSat1(t,4:6))'-Tdot*THillECI*(vecSat3(t,1:3)-vecSat1(t,1:3))'; 
    vecdist14Hill(t,4:6)  = THillECI*(vecSat4(t,4:6)-vecSat1(t,4:6))'-Tdot*THillECI*(vecSat4(t,1:3)-vecSat1(t,1:3))';
    vecdist15Hill(t,4:6)  = THillECI*(vecSat5(t,4:6)-vecSat1(t,4:6))'-Tdot*THillECI*(vecSat5(t,1:3)-vecSat1(t,1:3))';
    
    vecdist12Hill(t,4:6)  = THillECI*(vecSat2(t,4:6)-vecSat1(t,4:6))'-THillECI*Tdot*THillECI*(vecSat2(t,1:3)-vecSat1(t,1:3))';
    vecdist13Hill(t,4:6)  = THillECI*(vecSat3(t,4:6)-vecSat1(t,4:6))'-THillECI*Tdot*THillECI*(vecSat3(t,1:3)-vecSat1(t,1:3))'; 
    vecdist14Hill(t,4:6)  = THillECI*(vecSat4(t,4:6)-vecSat1(t,4:6))'-THillECI*Tdot*THillECI*(vecSat4(t,1:3)-vecSat1(t,1:3))';
    vecdist15Hill(t,4:6)  = THillECI*(vecSat5(t,4:6)-vecSat1(t,4:6))'-THillECI*Tdot*THillECI*(vecSat5(t,1:3)-vecSat1(t,1:3))';
    
    vecdist23Hill(t,4:6)  = THillECI*(vecSat3(t,4:6)-vecSat2(t,4:6))'-THillECI*Tdot*THillECI*(vecSat3(t,1:3)-vecSat2(t,1:3))';
    vecdist34Hill(t,4:6)  = THillECI*(vecSat4(t,4:6)-vecSat3(t,4:6))'-THillECI*Tdot*THillECI*(vecSat4(t,1:3)-vecSat3(t,1:3))'; 
    vecdist45Hill(t,4:6)  = THillECI*(vecSat5(t,4:6)-vecSat4(t,4:6))'-THillECI*Tdot*THillECI*(vecSat5(t,1:3)-vecSat4(t,1:3))';
    vecdist52Hill(t,4:6)  = THillECI*(vecSat2(t,4:6)-vecSat5(t,4:6))'-THillECI*Tdot*THillECI*(vecSat2(t,1:3)-vecSat5(t,1:3))';
    
    vecdist65Hill(t,4:6)  = THillECI*(vecSat5(t,4:6)-vecSat6(t,4:6))'-THillECI*Tdot*THillECI*(vecSat5(t,1:3)-vecSat6(t,1:3))'; 
    vecdist64Hill(t,4:6)  = THillECI*(vecSat4(t,4:6)-vecSat6(t,4:6))'-THillECI*Tdot*THillECI*(vecSat4(t,1:3)-vecSat6(t,1:3))';
    vecdist63Hill(t,4:6)  = THillECI*(vecSat3(t,4:6)-vecSat6(t,4:6))'-THillECI*Tdot*THillECI*(vecSat3(t,1:3)-vecSat6(t,1:3))';
    vecdist62Hill(t,4:6)  = THillECI*(vecSat2(t,4:6)-vecSat6(t,4:6))'-THillECI*Tdot*THillECI*(vecSat2(t,1:3)-vecSat6(t,1:3))'; 
    
    % Determine Relative Velocities
    
    v12test = vecSat2(t,4:6)-vecSat1(t,4:6)- cross(n*z1,(vecSat2(t,1:3)-vecSat1(t,1:3))');
    v12real = THillECI*v12test';
    vecdist12Hill(t,4:6)  = THillECI*(vecSat2(t,4:6)-vecSat1(t,4:6))'+Tdot*(vecSat2(t,1:3)-vecSat1(t,1:3))';
    vecdist13Hill(t,4:6)  = THillECI*(vecSat3(t,4:6)-vecSat1(t,4:6))'+Tdot*(vecSat3(t,1:3)-vecSat1(t,1:3))'; 
    vecdist14Hill(t,4:6)  = THillECI*(vecSat4(t,4:6)-vecSat1(t,4:6))'+Tdot*(vecSat4(t,1:3)-vecSat1(t,1:3))';
    vecdist15Hill(t,4:6)  = THillECI*(vecSat5(t,4:6)-vecSat1(t,4:6))'+Tdot*(vecSat5(t,1:3)-vecSat1(t,1:3))';
    
    vecdist23Hill(t,4:6)  = THillECI*(vecSat3(t,4:6)-vecSat2(t,4:6))'+Tdot*(vecSat3(t,1:3)-vecSat2(t,1:3))';
    vecdist34Hill(t,4:6)  = THillECI*(vecSat4(t,4:6)-vecSat3(t,4:6))'+Tdot*(vecSat4(t,1:3)-vecSat3(t,1:3))'; 
    vecdist45Hill(t,4:6)  = THillECI*(vecSat5(t,4:6)-vecSat4(t,4:6))'+Tdot*(vecSat5(t,1:3)-vecSat4(t,1:3))';
    vecdist52Hill(t,4:6)  = THillECI*(vecSat2(t,4:6)-vecSat5(t,4:6))'+Tdot*(vecSat2(t,1:3)-vecSat5(t,1:3))';
    
    vecdist65Hill(t,4:6)  = THillECI*(vecSat5(t,4:6)-vecSat6(t,4:6))'+Tdot*(vecSat5(t,1:3)-vecSat6(t,1:3))'; 
    vecdist64Hill(t,4:6)  = THillECI*(vecSat4(t,4:6)-vecSat6(t,4:6))'+Tdot*(vecSat4(t,1:3)-vecSat6(t,1:3))';
    vecdist63Hill(t,4:6)  = THillECI*(vecSat3(t,4:6)-vecSat6(t,4:6))'+Tdot*(vecSat3(t,1:3)-vecSat6(t,1:3))';
    vecdist62Hill(t,4:6)  = THillECI*(vecSat2(t,4:6)-vecSat6(t,4:6))'+Tdot*(vecSat2(t,1:3)-vecSat6(t,1:3))'; 
    
    vecSat1Hill(t,1:3) = THillECI*(vecSat1(t,1:3)-posCenterECI)';
    vecSat1Hill(t,4:6) = THillECI*(vecSat1(t,4:6)-velCenterECI)'+Tdot*(vecSat1(t,1:3)-posCenterECI)'; 
    
    vecSat2Hill(t,1:3) = THillECI*(vecSat2(t,1:3)-posCenterECI)';
    vecSat2Hill(t,4:6) = THillECI*(vecSat2(t,4:6)-velCenterECI)'+Tdot*(vecSat2(t,1:3)-posCenterECI)'; 
    
    vecSat3Hill(t,1:3) = THillECI*(vecSat3(t,1:3)-posCenterECI)';
    vecSat3Hill(t,4:6) = THillECI*(vecSat3(t,4:6)-velCenterECI)'+Tdot*(vecSat3(t,1:3)-posCenterECI)'; 
    
    vecSat4Hill(t,1:3) = THillECI*(vecSat4(t,1:3)-posCenterECI)';
    vecSat4Hill(t,4:6) = THillECI*(vecSat4(t,4:6)-velCenterECI)'+Tdot*(vecSat4(t,1:3)-posCenterECI)'; 
    
    vecSat5Hill(t,1:3) = THillECI*(vecSat5(t,1:3)-posCenterECI)';
    vecSat5Hill(t,4:6) = THillECI*(vecSat5(t,4:6)-velCenterECI)'+Tdot*(vecSat5(t,1:3)-posCenterECI)'; 
    
    vecSat6Hill(t,1:3) = THillECI*(vecSat6(t,1:3)-posCenterECI)';
    vecSat6Hill(t,4:6) = THillECI*(vecSat6(t,4:6)-velCenterECI)'+Tdot*(vecSat6(t,1:3)-posCenterECI)'; 
    
    THillECIArray(3*(t-1)+1:3*t,:) = THillECI;
    THillECIAnArray(3*(t-1)+1:3*t,:) = THillECIAn; 
    TdotArray(3*(t-1)+1:3*t,:) = Tdot;
    SArray(3*(t-1)+1:3*t,:) = S;
    nArray(t) = n;
    xArray(t,:) = x;
    yArray(t,:) = y;
    zArray(t,:) = z;
    x1Array(t,:) = x1;
    y1Array(t,:) = y1;
    z1Array(t,:) = z1;
    z2Array(t,:) = z2;
    argLatArray(t) = argLat;
    TtotalArray(6*(t-1)+1:6*t,:) = Ttotal;
    %vRef = Tdot*THillECI*obj.centerOffset';
end
%}
%{
figure
sgtitle("RotationMatrix ECI-Hill")
subplot(3,3,1)
plot(THillECIArray(1:3:end,1))
hold on
plot(THillECIAnArray(1:3:end,1))

ylim([-1 1])

subplot(3,3,2) 
plot(THillECIArray(1:3:end,2))
hold on
plot(THillECIAnArray(1:3:end,2))
ylim([-1 1])

subplot(3,3,3)
plot(THillECIArray(1:3:end,3))
hold on
plot(THillECIAnArray(1:3:end,3))
ylim([-1 1])

subplot(3,3,4)
plot(THillECIArray(2:3:end,1))
hold on
plot(THillECIAnArray(2:3:end,1))
ylim([-1 1])

subplot(3,3,5)
plot(THillECIArray(2:3:end,2))
hold on
plot(THillECIAnArray(2:3:end,2))
ylim([-1 1])

subplot(3,3,6)
plot(THillECIArray(2:3:end,3))
hold on
plot(THillECIAnArray(2:3:end,3))
ylim([-1 1])

subplot(3,3,7)
plot(THillECIArray(3:3:end,1))
hold on
plot(THillECIAnArray(3:3:end,1))
ylim([-1 1])

subplot(3,3,8)
plot(THillECIArray(3:3:end,2))
hold on
plot(THillECIAnArray(3:3:end,2))
ylim([-1 1])

subplot(3,3,9)
plot(THillECIArray(3:3:end,3))
hold on
plot(THillECIAnArray(3:3:end,3))
ylim([-1 1])

figure
sgtitle("Difference in RotationMatrix ECI-Hill")
subplot(3,3,1)
plot(THillECIArray(1:3:end,1)-THillECIAnArray(1:3:end,1))

subplot(3,3,2) 
plot(THillECIArray(1:3:end,2)-THillECIAnArray(1:3:end,2))

subplot(3,3,3)
plot(THillECIArray(1:3:end,3)-THillECIAnArray(1:3:end,3))

subplot(3,3,4)
plot(THillECIArray(2:3:end,1)-THillECIAnArray(2:3:end,1))


subplot(3,3,5)
plot(THillECIArray(2:3:end,2)-THillECIAnArray(2:3:end,2))


subplot(3,3,6)
plot(THillECIArray(2:3:end,3)-THillECIAnArray(2:3:end,3))

subplot(3,3,7)
plot(THillECIArray(3:3:end,1)-THillECIAnArray(3:3:end,1))

subplot(3,3,8)
plot(THillECIArray(3:3:end,2)-THillECIAnArray(3:3:end,2))

subplot(3,3,9)
plot(THillECIArray(3:3:end,3)-THillECIAnArray(3:3:end,3))

figure
sgtitle("Rotation Matrix Tdot")
subplot(3,3,1)
plot(TdotArray(1:3:end,1))
hold on;
plot(diff(THillECIArray(1:3:end,1)))
plot(diff(THillECIAnArray(1:3:end,1)))

subplot(3,3,2)
plot(TdotArray(1:3:end,2))
hold on;
plot(diff(THillECIArray(1:3:end,2)))
plot(diff(THillECIAnArray(1:3:end,2)))

subplot(3,3,3)
plot(TdotArray(1:3:end,3))
hold on;
plot(diff(THillECIArray(1:3:end,3)))
plot(diff(THillECIAnArray(1:3:end,3)))

subplot(3,3,4)
plot(TdotArray(2:3:end,1))
hold on;
plot(diff(THillECIArray(2:3:end,1)))
plot(diff(THillECIAnArray(2:3:end,1)))

subplot(3,3,5)
plot(TdotArray(2:3:end,2))
hold on;
plot(diff(THillECIArray(2:3:end,2)))
plot(diff(THillECIAnArray(2:3:end,2)))

subplot(3,3,6)
plot(TdotArray(2:3:end,3))
hold on;
plot(diff(THillECIArray(2:3:end,3)))
plot(diff(THillECIAnArray(2:3:end,3)))

subplot(3,3,7)
plot(TdotArray(3:3:end,1))
hold on;
plot(diff(THillECIArray(3:3:end,1)))
plot(diff(THillECIAnArray(3:3:end,1)))

subplot(3,3,8)
plot(TdotArray(3:3:end,2))
hold on;
plot(diff(THillECIArray(3:3:end,2)))
plot(diff(THillECIAnArray(3:3:end,2)))

subplot(3,3,9)
plot(TdotArray(3:3:end,3))
hold on;
plot(diff(THillECIArray(3:3:end,3)))
plot(diff(THillECIAnArray(3:3:end,3)))
%}
% Test Conversion Function
%{
[vecdist12HillFunc,vel12] = ECI2Hill_Vectorized(vecSat1(:,1:3)',vecSat1(:,4:6)',vecSat2(:,1:3)',vecSat2(:,4:6)');
[vecdist13HillFunc,vel13] = ECI2Hill_Vectorized(vecSat1(:,1:3)',vecSat1(:,4:6)',vecSat3(:,1:3)',vecSat3(:,4:6)');
[vecdist14HillFunc,vel14] = ECI2Hill_Vectorized(vecSat1(:,1:3)',vecSat1(:,4:6)',vecSat4(:,1:3)',vecSat4(:,4:6)');
[vecdist15HillFunc,vel15] = ECI2Hill_Vectorized(vecSat1(:,1:3)',vecSat1(:,4:6)',vecSat5(:,1:3)',vecSat5(:,4:6)');

[vecdist23HillFunc,vel23] = ECI2Hill_Vectorized(vecSat2(:,1:3)',vecSat2(:,4:6)',vecSat3(:,1:3)',vecSat3(:,4:6)');
[vecdist34HillFunc,vel34] = ECI2Hill_Vectorized(vecSat3(:,1:3)',vecSat3(:,4:6)',vecSat4(:,1:3)',vecSat4(:,4:6)');
[vecdist45HillFunc,vel45] = ECI2Hill_Vectorized(vecSat4(:,1:3)',vecSat4(:,4:6)',vecSat5(:,1:3)',vecSat5(:,4:6)');
[vecdist52HillFunc,vel52] = ECI2Hill_Vectorized(vecSat5(:,1:3)',vecSat5(:,4:6)',vecSat2(:,1:3)',vecSat2(:,4:6)');

[vecdist65HillFunc,vel65] = ECI2Hill_Vectorized(vecSat6(:,1:3)',vecSat6(:,4:6)',vecSat5(:,1:3)',vecSat5(:,4:6)');
[vecdist64HillFunc,vel64] = ECI2Hill_Vectorized(vecSat6(:,1:3)',vecSat6(:,4:6)',vecSat4(:,1:3)',vecSat4(:,4:6)');
[vecdist63HillFunc,vel63] = ECI2Hill_Vectorized(vecSat6(:,1:3)',vecSat6(:,4:6)',vecSat3(:,1:3)',vecSat3(:,4:6)');
[vecdist62HillFunc,vel62] = ECI2Hill_Vectorized(vecSat6(:,1:3)',vecSat6(:,4:6)',vecSat2(:,1:3)',vecSat2(:,4:6)');
vecdist12HillFunc = [vecdist12HillFunc;vel12]';
vecdist13HillFunc = [vecdist13HillFunc;vel13]';
vecdist14HillFunc = [vecdist14HillFunc;vel14]';
vecdist15HillFunc = [vecdist15HillFunc;vel15]';

vecdist23HillFunc = [vecdist23HillFunc;vel23]';
vecdist34HillFunc = [vecdist34HillFunc;vel34]';
vecdist45HillFunc = [vecdist45HillFunc;vel45]';
vecdist52HillFunc = [vecdist52HillFunc;vel52]';

vecdist65HillFunc = [vecdist65HillFunc;vel65]';
vecdist64HillFunc = [vecdist64HillFunc;vel64]';
vecdist63HillFunc = [vecdist63HillFunc;vel63]';
vecdist62HillFunc = [vecdist62HillFunc;vel62]';

figure;
sgtitle("Conversion between ECI and Hill Frame");
subplot(2,3,1)
plot(vecdist12HillFunc(:,1),'g')
hold on
plot(vecdist12Hill(:,1),'r')

subplot(2,3,2)
plot(vecdist12HillFunc(:,2),'g')
hold on
plot(vecdist12Hill(:,2),'r')

subplot(2,3,3)
plot(vecdist12HillFunc(:,3),'g')
hold on
plot(vecdist12Hill(:,3),'r')

subplot(2,3,4)
plot(vecdist12HillFunc(:,4),'g')
hold on
plot(vecdist12Hill(:,4),'r')

subplot(2,3,5)
plot(vecdist12HillFunc(:,5),'g')
hold on
plot(vecdist12Hill(:,5),'r')

subplot(2,3,6)
plot(vecdist12HillFunc(:,6),'g')
hold on
plot(vecdist12Hill(:,6),'r')

%%
% Determine location in Formation Frame
FFvec = (vecSat1 + vecSat2 + vecSat3 + vecSat4 + vecSat5 + vecSat6)/6; %location of Formation Frame
FFloc = (posSat1 + posSat2 + posSat3 + posSat4 + posSat5 + posSat6)/6; %location of Formation Frame
posSat1FF = quatrotate(quatinv(formationOrientation),posSat1-FFloc);
posSat2FF = quatrotate(quatinv(formationOrientation),posSat2-FFloc);
posSat3FF = quatrotate(quatinv(formationOrientation),posSat3-FFloc);
posSat4FF = quatrotate(quatinv(formationOrientation),posSat4-FFloc);
posSat5FF = quatrotate(quatinv(formationOrientation),posSat5-FFloc);
posSat6FF = quatrotate(quatinv(formationOrientation),posSat6-FFloc);

% Relative position measurements
tSettle = 0;
s11 = relMeasOutput(1:24:end,:); % Distance 1x2
s12 = relMeasOutput(2:24:end,:); % Distance 1x3
s13 = relMeasOutput(3:24:end,:); % Distance 1x4
s14 = relMeasOutput(4:24:end,:); % Distance 1x5

s21 = relMeasOutput(5:24:end,:); % Distance 2x1
s22 = relMeasOutput(6:24:end,:); % Distance 2x5
s23 = relMeasOutput(7:24:end,:); % Distance 2x6
s24 = relMeasOutput(8:24:end,:); % Distance 2x3

s31 = relMeasOutput(9:24:end,:);  % Distance 3x1
s32 = relMeasOutput(10:24:end,:); % Distance 3x2
s33 = relMeasOutput(11:24:end,:); % Distance 3x6
s34 = relMeasOutput(12:24:end,:); % Distance 3x4

s41 = relMeasOutput(13:24:end,:); % Distance 4x3
s42 = relMeasOutput(14:24:end,:); % Distance 4x6
s43 = relMeasOutput(15:24:end,:); % Distance 4x5
s44 = relMeasOutput(16:24:end,:); % Distance 4x4

s51 = relMeasOutput(17:24:end,:); % Distance 5x4
s52 = relMeasOutput(18:24:end,:); % Distance 5x6
s53 = relMeasOutput(19:24:end,:); % Distance 5x2
s54 = relMeasOutput(20:24:end,:); % Distance 5x1

s61 = relMeasOutput(21:24:end,:); % Distance 6x5
s62 = relMeasOutput(22:24:end,:); % Distance 6x4
s63 = relMeasOutput(23:24:end,:); % Distance 6x3
s64 = relMeasOutput(24:24:end,:); % Distance 6x2

sStack = [s11,s12,s13,s14,...
          s21,s22,s23,s24,...
          s31,s32,s33,s34,...
          s41,s42,s43,s44,...
          s51,s52,s53,s54,...
          s61,s62,s63,s64];
% Relative distances in Hill frame
s11Hill = relHillOutput(1:24:end,:); % Distance 1x2
s12Hill = relHillOutput(2:24:end,:); % Distance 1x3
s13Hill = relHillOutput(3:24:end,:); % Distance 1x4
s14Hill = relHillOutput(4:24:end,:); % Distance 1x5

s21Hill = relHillOutput(5:24:end,:); % Distance 2x1
s22Hill = relHillOutput(6:24:end,:); % Distance 2x5
s23Hill = relHillOutput(7:24:end,:); % Distance 2x6
s24Hill = relHillOutput(8:24:end,:); % Distance 2x3

s31Hill = relHillOutput(9:24:end,:);  % Distance 3x1
s32Hill = relHillOutput(10:24:end,:); % Distance 3x2
s33Hill = relHillOutput(11:24:end,:); % Distance 3x6
s34Hill = relHillOutput(12:24:end,:); % Distance 3x4

s41Hill = relHillOutput(13:24:end,:); % Distance 4x3
s42Hill = relHillOutput(14:24:end,:); % Distance 4x6
s43Hill = relHillOutput(15:24:end,:); % Distance 4x5
s44Hill = relHillOutput(16:24:end,:); % Distance 4x4

s51Hill = relHillOutput(17:24:end,:); % Distance 5x4
s52Hill = relHillOutput(18:24:end,:); % Distance 5x6
s53Hill = relHillOutput(19:24:end,:); % Distance 5x2
s54Hill = relHillOutput(20:24:end,:); % Distance 5x1

s61Hill = relHillOutput(21:24:end,:); % Distance 6x5
s62Hill = relHillOutput(22:24:end,:); % Distance 6x4
s63Hill = relHillOutput(23:24:end,:); % Distance 6x3
s64Hill = relHillOutput(24:24:end,:); % Distance 6x2

sHillStack = [s11Hill,s12Hill,s13Hill,s14Hill,...
              s21Hill,s22Hill,s23Hill,s24Hill,...
              s31Hill,s32Hill,s33Hill,s34Hill,...
              s41Hill,s42Hill,s43Hill,s44Hill,...
              s51Hill,s52Hill,s53Hill,s54Hill,...
              s61Hill,s62Hill,s63Hill,s64Hill];
% Estimated position in formation frame
% Ordering of distances
% [s12,s13,s14,s15,s23,s34,s45,s52,s65,s64,s63,s62]
% Estimated distances
distEst12 = (s11-s21)/2;
distEst13 = (s12-s31)/2;
distEst14 = (s13-s44)/2;
distEst15 = (s14-s54)/2;

distEst23 = (s24-s32)/2;
distEst34 = (s34-s41)/2;
distEst45 = (s43-s51)/2;
distEst52 = (s53-s22)/2;

distEst65 = (s61-s52)/2;
distEst64 = (s62-s42)/2;
distEst63 = (s63-s33)/2;
distEst62 = (s64-s23)/2;

% Errors from one measurement
errOM11 = vecdist11 - s11;
errOM12 = vecdist12 - s12;
errOM13 = vecdist13 - s13;
errOM14 = vecdist14 - s14;

errOM21 = vecdist21 - s21;
errOM22 = vecdist22 - s22;
errOM23 = vecdist23 - s23;
errOM24 = vecdist24 - s24;

errOM31 = vecdist31 - s31;
errOM32 = vecdist32 - s32;
errOM33 = vecdist33 - s33;
errOM34 = vecdist34 - s34;

errOM41 = vecdist41 - s41;
errOM42 = vecdist42 - s42;
errOM43 = vecdist43 - s43;
errOM44 = vecdist44 - s44;

errOM51 = vecdist51 - s51;
errOM52 = vecdist52 - s52;
errOM53 = vecdist53 - s53;
errOM54 = vecdist54 - s54;

errOM61 = vecdist61 - s61;
errOM62 = vecdist62 - s62;
errOM63 = vecdist63 - s63;
errOM64 = vecdist64 - s64;

% Errors from two measurements
errTM12 = vecdist11 - dist12;
errTM13 = vecdist13 - dist13;
errTM14 = vecdist11 - dist12;
errTM15 = vecdist13 - dist13;

errTM23 = vecdist24 - dist23;
errTM34 = vecdist34 - dist34;
errTM45 = vecdist43 - dist45;
errTM52 = vecdist53 - dist52;

errTM65 = vecdist61 - dist65;
errTM64 = vecdist62 - dist64;
errTM63 = vecdist63 - dist63;
errTM62 = vecdist64 - dist62;
% Error of centerpoint from four measurements
FFloc1 = vecSatMeas1(:,1:3) + (s11(:,1:3)+s12(:,1:3)+s13(:,1:3)+s14(:,1:3))/4;
vFFloc1 = vecSatMeas1(:,4:6) + (s11(:,4:6)+s12(:,4:6)+s13(:,4:6)+s14(:,4:6))/4;
FFloc2 = vecSatMeas2(:,1:3) + (s21(:,1:3)+s22(:,1:3)+s23(:,1:3)+s24(:,1:3))/4;
FFloc3 = vecSatMeas3(:,1:3) + (s31(:,1:3)+s32(:,1:3)+s33(:,1:3)+s34(:,1:3))/4;
FFloc4 = vecSatMeas4(:,1:3) + (s41(:,1:3)+s42(:,1:3)+s43(:,1:3)+s44(:,1:3))/4;
FFloc5 = vecSatMeas5(:,1:3) + (s51(:,1:3)+s52(:,1:3)+s53(:,1:3)+s54(:,1:3))/4;
FFloc6 = vecSatMeas6(:,1:3) + (s61(:,1:3)+s62(:,1:3)+s63(:,1:3)+s64(:,1:3))/4;

errCenter1 = (FFloc) - FFloc1;
errCenter2 = (FFloc) - FFloc2;
errCenter3 = (FFloc) - FFloc3;
errCenter4 = (FFloc) - FFloc4;
errCenter5 = (FFloc) - FFloc5;
errCenter6 = (FFloc) - FFloc6;

% Error of centerpoint using all absolute measurements
FFlocRelMeas = 1/6*(FFloc1 + FFloc2 + FFloc3 + FFloc4 + FFloc5 + FFloc6);
FFlocAbsMeas = (vecSatMeas1(:,1:3) + vecSatMeas2(:,1:3) + ...
             vecSatMeas3(:,1:3) + vecSatMeas4(:,1:3) + ...
             vecSatMeas5(:,1:3) + vecSatMeas6(:,1:3))/6;
errCenterAll = FFloc - FFlocAbsMeas;
errCenterRel = FFloc - FFlocRelMeas;


%% Test A state matrix

A = sFormation.spacecraftArray(1).A;
Bf = sFormation.spacecraftArray(1).B;
B = [0,0,0;
     0,0,0;
     0,0,0;
     1,0,0;
     0,1,0;
     0,0,1];
zerosB = zeros(size(B));
Bformation = [-B    , B     , zerosB, zerosB, zerosB, zerosB;
              -B    , zerosB, B     , zerosB, zerosB, zerosB;
              -B    , zerosB, zerosB, B     , zerosB, zerosB;
              -B    , zerosB, zerosB, zerosB, B     , zerosB;
              zerosB,-B     , B     , zerosB, zerosB, zerosB;
              zerosB, zerosB,-B     , B     , zerosB, zerosB;
              zerosB, zerosB, zerosB, -B    , B     , zerosB;
              zerosB, B     , zerosB, zerosB,-B     , zerosB;
              zerosB, zerosB, zerosB, zerosB, B     ,-B;
              zerosB, zerosB, zerosB, B     , zerosB,-B;
              zerosB, zerosB, B     , zerosB, zerosB,-B;
              zerosB, B     , zerosB, zerosB, zerosB,-B];

state = [vecdist12HillFunc,vecdist13HillFunc,vecdist14HillFunc,vecdist15HillFunc,....
         vecdist23HillFunc,vecdist34HillFunc,vecdist45HillFunc,vecdist52HillFunc,....
         vecdist65HillFunc,vecdist64HillFunc,vecdist63HillFunc,vecdist62HillFunc];
     
test = (A*state(1:end-1,:)')'-state(2:end,:);%+Bf(:,1:18)*commandStack(2:end,:)')'
a = (A*state(1:end-1,:)')';
b = (Bf(:,1:18)*commandStack(1:end-1,:)')';
figure;
sgtitle("Linear Model One-step prediction error")
subplot(1,2,1)
plot(test(:,1:3))
subplot(1,2,2)
plot(cumsum(test(:,4:6)+b(:,4:6)))
hold on
%plot(b(:,4:6))
%%
linModel = zeros(size(state));
linModel(1,:) = state(1,:);


for ii = 2:steps
    linModel(ii,:) = A*linModel(ii-1,:)'+Bf(:,1:18)*commandStack(ii-1,:)';
end
figure;
sgtitle("Linear Model vs actual dynamics");
subplot(3,1,1)
plot(linModel(:,1),'r');
hold on;
plot(state(:,1),'g');

subplot(3,1,2)
plot(linModel(:,2),'r');
hold on;
plot(state(:,2),'g');

subplot(3,1,3)
plot(linModel(:,3),'r');
hold on;
plot(state(:,3),'g');
%%  Test nonlinear dynamics model

B = sFormation.spacecraftArray(1).B;
model = zeros(size(state));
model(1,:) = state(1,:);
modelErr = zeros(size(state));
for ii = 2:steps
    formCenter = FFloc1(ii-1,:);
    velCenter = vFFloc1(ii-1,:);
    [a,e,~,~,~,~,~,argLat] = rv2orb(formCenter',velCenter');
    model(ii,:) = state(ii-1,:) + relativeDynNonlinear(model(ii-1,:),formCenter,velCenter,a,e,argLat) + (B(1:72,1:18)*commandStack(ii-1,:)')';
    modelErr(ii,:) = model(ii,:) - state(ii,:);
end
figure;
sgtitle("Non Linear Model One-step prediction error")
subplot(1,2,1)
plot(modelErr(:,1:3))
subplot(1,2,2)
plot(modelErr(:,4:6))
%}
%% Relative Distance plot
width = 1.;
%{
xLimit = steps;
%
tSettle = 2;
time = tSettle:dt:tE;
figure;
sgtitle("Relative Motion in ECI")
% Satellite 1
subplot(2,3,1)
plot(time, dist12(tSettle:end),'k')
hold on;
plot(time, dist13(tSettle:end),'r')
plot(time, dist14(tSettle:end),'g')
plot(time, dist15(tSettle:end),'b')
%plot(time, dist16)
plot([time(tSettle-1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 1 [m]')

% Satellite 2
subplot(2,3,2)
plot(time, dist21(tSettle:end),'k')
hold on;
plot(time, dist25(tSettle:end),'r')
plot(time, dist26(tSettle:end),'g')
plot(time, dist23(tSettle:end),'b')
%plot(time, dist16)
plot([time(1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 2 [m]')

% Satellite 3
subplot(2,3,3)
plot(time, dist31(tSettle:end),'k')
hold on;
plot(time, dist32(tSettle:end),'r')
plot(time, dist36(tSettle:end),'g')
plot(time, dist34(tSettle:end),'b')

%plot(time, dist16)
plot([time(tSettle-1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 3 [m]')

% Satellite 4
subplot(2,3,4)
plot(time, dist43(tSettle:end),'k')
hold on;
plot(time, dist46(tSettle:end),'r')
plot(time, dist45(tSettle:end),'g')
plot(time, dist41(tSettle:end),'b')

plot([time(tSettle-1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 4 [m]')

% Satellite 5
subplot(2,3,5)
plot(time, dist54(tSettle:end),'k')
hold on;
plot(time, dist56(tSettle:end),'r')
plot(time, dist52(tSettle:end),'g')
plot(time, dist51(tSettle:end),'b')

%plot(time, dist16)
plot([time(tSettle-1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 5 [m]')

% Satellite 6
subplot(2,3,6)
plot(time, dist65(tSettle:end),'k')
hold on;
plot(time, dist64(tSettle:end),'r')
plot(time, dist63(tSettle:end),'g')
plot(time, dist62(tSettle:end),'b')

%plot(time, dist16)
plot([time(tSettle-1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 6 [m]')
%}
%% PLOTS Measurement vs Physical relative distances
%{
figure
sgtitle("Measurement of relative Motion and true relative Motion")
subplot(3,4,1)
plot(vecdist11(:,1:3))

subplot(3,4,2)
plot(vecdist12(:,1:3))

subplot(3,4,3)
plot(vecdist13(:,1:3))
% 

subplot(3,4,1)
plot(s12(:,1))
hold on
plot(vecdist12(:,1))

subplot(3,1,2)
plot(s12(:,2))
hold on
plot(vecdist12(:,2))

subplot(3,1,3)
plot(s12(:,3))
hold on
plot(vecdist12(:,3))
% 
figure
subplot(3,1,1)
plot(s13(:,1))
hold on
plot(vecdist13(:,1))

subplot(3,1,2)
plot(s13(:,2))
hold on
plot(vecdist13(:,2))

subplot(3,1,3)
plot(s13(:,3))
hold on
plot(vecdist13(:,3))
% 
figure
subplot(3,1,1)
plot(s14(:,1))
hold on
plot(vecdist14(:,1))

subplot(3,1,2)
plot(s14(:,2))
hold on
plot(vecdist14(:,2))

subplot(3,1,3)
plot(s14(:,3))
hold on
plot(vecdist14(:,3))
%}
%% Plot relative distances - measurement in Hill frame
%{
figure
sgtitle("True Relative distances in Hill frame")
subplot(3,4,1)
plot(vecdist12Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])

subplot(3,4,2)
plot(vecdist13Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])

subplot(3,4,3)
plot(vecdist14Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])

subplot(3,4,4)
plot(vecdist15Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])

subplot(3,4,5)
plot(vecdist23Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])

subplot(3,4,6)
plot(vecdist34Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])

subplot(3,4,7)
plot(vecdist45Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])

subplot(3,4,8)
plot(vecdist52Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])

subplot(3,4,9)
plot(vecdist65Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])

subplot(3,4,10)
plot(vecdist64Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])

subplot(3,4,11)
plot(vecdist63Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])

subplot(3,4,12)
plot(vecdist62Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])
%}
%% Kalman vs Real 
%{
width = 1;

yLimit = 1;
figure('units','normalized','outerposition',[0 0 1 1])
sgtitle("Kalman Filter Error");
subplot(3,4,1)
plot(vecdist12Hill(:,1:3) - kalmanOutput(:,(1-1)*6+1:(1-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,2)
plot(vecdist13Hill(:,1:3) - kalmanOutput(:,(2-1)*6+1:(2-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,3)
plot(vecdist14Hill(:,1:3) - kalmanOutput(:,(3-1)*6+1:(3-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,4)
plot(vecdist15Hill(:,1:3) - kalmanOutput(:,(4-1)*6+1:(4-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,5)
plot(vecdist23Hill(:,1:3) - kalmanOutput(:,(5-1)*6+1:(5-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,6)
plot(vecdist34Hill(:,1:3) - kalmanOutput(:,(6-1)*6+1:(6-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,7)
plot(vecdist45Hill(:,1:3) - kalmanOutput(:,(7-1)*6+1:(7-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,8)
plot(vecdist52Hill(:,1:3) - kalmanOutput(:,(8-1)*6+1:(8-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,9)
plot(vecdist65Hill(:,1:3) - kalmanOutput(:,(9-1)*6+1:(9-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,10)
plot(vecdist64Hill(:,1:3) - kalmanOutput(:,(10-1)*6+1:(10-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,11)
plot(vecdist63Hill(:,1:3) - kalmanOutput(:,(11-1)*6+1:(11-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,12)
plot(vecdist62Hill(:,1:3) - kalmanOutput(:,(12-1)*6+1:(12-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
%}
%% Plot true distance in hill - measurement
%{
yLimit = 0.2;
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,4,1)

plot(vecdist12Hill(:,1:3) - s11Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,2)
plot(vecdist13Hill(:,1:3) - s12Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,3)
plot(vecdist14Hill(:,1:3) - s13Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,4)
plot(vecdist15Hill(:,1:3) - s14Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,5)
plot(vecdist23Hill(:,1:3) - s24Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,6)
plot(vecdist34Hill(:,1:3) - s34Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,7)
plot(vecdist45Hill(:,1:3) - s43Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,8)
plot(vecdist52Hill(:,1:3) - s53Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,9)
plot(vecdist65Hill(:,1:3) - s61Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,10)
plot(vecdist64Hill(:,1:3) - s62Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,11)
plot(vecdist63Hill(:,1:3) - s63Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])

subplot(3,4,12)
plot(vecdist62Hill(:,1:3) - s64Hill(:,1:3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
%}
%% Plot output of Kalman Innovation
yLimit = 3;
xLimit = steps;
figure('units','normalized','outerposition',[0 0 1 1])
%sgtitle("Kalman Filter Innovation Vectors")
subplot(2,3,1)
plot(residualOutput(:,(1-1)*6+1:(1-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-2')  
ax = gca;
ax.FontSize = mediumSize; 

subplot(2,3,2)
plot(residualOutput(:,(2-1)*6+1:(2-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-3') 
ax = gca;
ax.FontSize = mediumSize; 

subplot(2,3,4)
plot(residualOutput(:,(3-1)*6+1:(3-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-4') 
ax = gca;
ax.FontSize = mediumSize; 


subplot(2,3,5)
linex = plot(residualOutput(:,(4-1)*6+1),"Linewidth",width);
hold on;
liney = plot(residualOutput(:,(4-1)*6+2),"Linewidth",width);
linez = plot(residualOutput(:,(4-1)*6+3),"Linewidth",width);
hL = legend([linex,liney,linez],{'Radial Component','Along-Track Component','Cross-Track Component'});
% Programatically move the Legend
newPosition = [0.65 0.5 0.2 0.2];
newUnits = 'normalized';
set(hL,'Position', newPosition,'Units', newUnits);
hl.FontSize = 18;
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-5') 
ax = gca;
ax.FontSize = mediumSize; 


%{
yLimit = 0.3;
xLimit = steps;
figure('units','normalized','outerposition',[0 0 1 1])
sgtitle("Kalman Filter Innovation Vectors")
subplot(3,4,1)
plot(residualOutput(:,(1-1)*6+1:(1-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-2')  
ax = gca;
ax.FontSize = mediumSize; 

subplot(3,4,2)
plot(residualOutput(:,(2-1)*6+1:(2-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-3') 
ax = gca;
ax.FontSize = mediumSize; 

subplot(3,4,3)
plot(residualOutput(:,(3-1)*6+1:(3-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-4') 
ax = gca;
ax.FontSize = mediumSize; 


subplot(3,4,4)
plot(residualOutput(:,(4-1)*6+1:(4-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-5') 
ax = gca;
ax.FontSize = mediumSize; 


subplot(3,4,5)
plot(residualOutput(:,(5-1)*6+1:(5-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 2-3') 
ax = gca;
ax.FontSize = mediumSize; 


subplot(3,4,6)
plot(residualOutput(:,(6-1)*6+1:(6-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 3-4') 
ax = gca;
ax.FontSize = mediumSize; 


subplot(3,4,7)
plot(residualOutput(:,(7-1)*6+1:(7-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 4-5') 
ax = gca;
ax.FontSize = mediumSize; 


subplot(3,4,8)
plot(residualOutput(:,(8-1)*6+1:(8-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 5-2') 
ax = gca;
ax.FontSize = mediumSize; 


subplot(3,4,9)
plot(residualOutput(:,(9-1)*6+1:(9-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 6-5') 
ax = gca;
ax.FontSize = mediumSize; 

subplot(3,4,10)
plot(residualOutput(:,(10-1)*6+1:(10-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 6-4') 
ax = gca;
ax.FontSize = mediumSize; 

subplot(3,4,11)
plot(residualOutput(:,(11-1)*6+1:(11-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 6-3') 
ax = gca;
ax.FontSize = mediumSize; 

subplot(3,4,12)
plot(residualOutput(:,(12-1)*6+1:(12-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 6-2') 
ax = gca;
ax.FontSize = mediumSize; 
%}
%% 

%{
yLimit = 0;
xLimit = steps;
figure('units','normalized','outerposition',[0 0 1 1])
sgtitle("Kalman Filter Innovation Vectors")
subplot(3,4,1)
plot(residualOutput(:,(1-1)*6+4:(1-1)*6+6),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m/s]')
title('Relative Velocity 1-2')  
ax = gca;
ax.FontSize = mediumSize; 

subplot(3,4,2)
plot(residualOutput(:,(2-1)*6+4:(2-1)*6+6),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m/s]')
title('Relative Velocity 1-3') 
ax = gca;
ax.FontSize = mediumSize; 

subplot(3,4,3)
plot(residualOutput(:,(3-1)*6+4:(3-1)*6+6),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m/s]')
title('Relative Velocity 1-4') 
ax = gca;
ax.FontSize = mediumSize; 


subplot(3,4,4)
plot(residualOutput(:,(4-1)*6+4:(4-1)*6+6),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m/s]')
title('Relative Velocity 1-5') 
ax = gca;
ax.FontSize = mediumSize; 


subplot(3,4,5)
plot(residualOutput(:,(5-1)*6+4:(5-1)*6+6),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m/s]')
title('Relative Velocity 2-3') 
ax = gca;
ax.FontSize = mediumSize; 


subplot(3,4,6)
plot(residualOutput(:,(6-1)*6+4:(6-1)*6+6),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m/s]')
title('Relative Velocity 3-4') 
ax = gca;
ax.FontSize = mediumSize; 


subplot(3,4,7)
plot(residualOutput(:,(7-1)*6+4:(7-1)*6+6),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m/s]')
title('Relative Velocity 4-5') 
ax = gca;
ax.FontSize = mediumSize; 


subplot(3,4,8)
plot(residualOutput(:,(8-1)*6+4:(8-1)*6+6),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m/s]')
title('Relative Velocity 5-2') 
ax = gca;
ax.FontSize = mediumSize; 


subplot(3,4,9)
plot(residualOutput(:,(9-1)*6+4:(9-1)*6+6),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m/s]')
title('Relative Velocity 6-5') 
ax = gca;
ax.FontSize = mediumSize; 

subplot(3,4,10)
plot(residualOutput(:,(10-1)*6+4:(10-1)*6+6),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m/s]')
title('Relative Velocity 6-4') 
ax = gca;
ax.FontSize = mediumSize; 

subplot(3,4,11)
plot(residualOutput(:,(11-1)*6+4:(11-1)*6+6),"Linewidth",width)
xlim([0 xLimit])
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m/s]')
title('Relative Velocity 6-3') 
ax = gca;
ax.FontSize = mediumSize; 

subplot(3,4,12)
plot(residualOutput(:,(12-1)*6+4:(12-1)*6+6),"Linewidth",width)
xlim([0 xLimit]) 
if yLimit>0
    ylim([-yLimit yLimit])
end
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m/s]')
title('Relative Velocity 6-2') 
ax = gca;
ax.FontSize = mediumSize; 

%}
%% Extended Kalman Innovations
%{
yLimit = 0.2;
xLimit = steps;
figure('units','normalized','outerposition',[0 0 1 1])
sgtitle("Extended Kalman Filter Innovation Vectors")
subplot(3,4,1)
plot(extResidualOutput(:,(1-1)*6+1:(1-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-2')  

subplot(3,4,2)
plot(extResidualOutput(:,(2-1)*6+1:(2-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-3') 

subplot(3,4,3)
plot(extResidualOutput(:,(3-1)*6+1:(3-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-4') 


subplot(3,4,4)
plot(extResidualOutput(:,(4-1)*6+1:(4-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-5') 


subplot(3,4,5)
plot(extResidualOutput(:,(5-1)*6+1:(5-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 2-3') 


subplot(3,4,6)
plot(extResidualOutput(:,(6-1)*6+1:(6-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 3-4') 


subplot(3,4,7)
plot(extResidualOutput(:,(7-1)*6+1:(7-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 4-5') 


subplot(3,4,8)
plot(extResidualOutput(:,(8-1)*6+1:(8-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 5-2') 


subplot(3,4,9)
plot(extResidualOutput(:,(9-1)*6+1:(9-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 6-5') 

subplot(3,4,10)
plot(extResidualOutput(:,(10-1)*6+1:(10-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 6-4') 

subplot(3,4,11)
plot(extResidualOutput(:,(11-1)*6+1:(11-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 6-3') 

subplot(3,4,12)
plot(extResidualOutput(:,(12-1)*6+1:(12-1)*6+3),"Linewidth",width)
xlim([0 xLimit]) 
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 6-2') 
%}
%% Plot Orbital Elements
%{
[aFF,eFF,incFF,OFF,oFF,MFF,trulonFF,argLatFF,lonPerFF] = rv2orb(center(:,1:3)',center(:,4:6)');

[a1,e1,inc1,O1,o1,M1,trulon1,argLat1,lonPer1] = rv2orb(vecSat1(:,1:3)',vecSat1(:,4:6)');
[a2,e2,inc2,O2,o2,M2,trulon2,argLat2,lonPer2]  = rv2orb(vecSat2(:,1:3)',vecSat2(:,4:6)');
[a3,e3,inc3,O3,o3,M3,trulon3,argLat3,lonPer3]  = rv2orb(vecSat3(:,1:3)',vecSat3(:,4:6)');
[a4,e4,inc4,O4,o4,M4,trulon4,argLat4,lonPer4]  = rv2orb(vecSat4(:,1:3)',vecSat4(:,4:6)');
[a5,e5,inc5,O5,o5,M5,trulon5,argLat5,lonPer5]  = rv2orb(vecSat5(:,1:3)',vecSat5(:,4:6)');
[a6,e6,inc6,O6,o6,M6,trulon6,argLat6,lonPer6]  = rv2orb(vecSat6(:,1:3)',vecSat6(:,4:6)');
figure
plot(0:dt:tE-1,[aFF',a1',a2',a3',a4',a5',a6'])
title('Semi-major Axis')
figure
plot(0:dt:tE-1,[eFF',e1',e2',e3',e4',e5',e6'])
title('Eccentricity')
figure
plot(0:dt:tE-1,[incFF',inc1',inc2',inc3',inc4',inc5',inc6'])
title('Inclination')
figure
plot(0:dt:tE-1,[OFF',O1',O2',O3',O4',O5',O6'])
title('RAAN')
figure
plot(0:dt:tE-1,[MFF',M1',M2',M3',M4',M5',M6'])
title('True Anomaly')
figure
plot(0:dt:tE-1,Constants.muEarth./aFF.^3);
title('Mean motion')
%}
%% Test residuals

alphLow = 0.0025;%(2*pi*Constants.dt*2/tE)/(2*pi*Constants.dt*2/tE + 1);%1.6941e-04;%
alphHigh = 0.99;
aHL = 0.8;
discLowPass = zeros(size(residualOutput));
discHighPass = zeros(size(residualOutput));
discLowPass(1,:) = residualOutput(1,:);
discHighPass(1,:) = residualOutput(1,:);
discHighLowPass = zeros(size(residualOutput));
discHighLowPass(1,:) = residualOutput(1,:);

for ii = 2:length(residualOutput)
    discLowPass(ii,:) = alphLow * residualOutput(ii,:) + (1-alphLow) * discLowPass(ii-1,:);
    discHighPass(ii,:) = alphHigh * discHighPass(ii-1,:) + alphHigh * (residualOutput(ii,:)-residualOutput(ii-1,:));
    discHighLowPass(ii,:) = aHL * discHighPass(ii,:) + (1-aHL) * discHighLowPass(ii-1,:);
end
%{
ylimit = 0.5
figure
sgtitle("Discrete Filters")
subplot(1,3,1)
plot(discLowPass(:,1))
xlabel("Time")
ylabel("Filtered Innovation")
ylim([-ylimit ylimit])
title("High Pass 1")

subplot(1,3,2)
plot(discLowPass(:,2))
xlabel("Time")
ylabel("Filtered Innovation")
ylim([-ylimit ylimit])
title("High Pass 2")

subplot(1,3,3)
plot(discLowPass(:,3))
xlabel("Time")
ylabel("Filtered Innovation")
ylim([-ylimit ylimit])

title("High Pass 3")
%}

resLP = residualOutput-discLowPass;
%{
yLimit = 0.2;
xLimit = steps;
figure('units','normalized','outerposition',[0 0 1 1])
sgtitle("Kalman Filter Innovation Vectors,high pass filtered")
subplot(3,4,1)
plot(resLP(:,(1-1)*6+1:(1-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-2')  

subplot(3,4,2)
plot(resLP(:,(2-1)*6+1:(2-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-3') 

subplot(3,4,3)
plot(resLP(:,(3-1)*6+1:(3-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-4') 


subplot(3,4,4)
plot(resLP(:,(4-1)*6+1:(4-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 1-5') 


subplot(3,4,5)
plot(resLP(:,(5-1)*6+1:(5-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 2-3') 


subplot(3,4,6)
plot(resLP(:,(6-1)*6+1:(6-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 3-4') 

subplot(3,4,7)
plot(resLP(:,(7-1)*6+1:(7-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 4-5') 


subplot(3,4,8)
plot(resLP(:,(8-1)*6+1:(8-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 5-2') 


subplot(3,4,9)
plot(resLP(:,(9-1)*6+1:(9-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 6-5') 

subplot(3,4,10)
plot(resLP(:,(10-1)*6+1:(10-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 6-4') 

subplot(3,4,11)
plot(resLP(:,(11-1)*6+1:(11-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 6-3') 

subplot(3,4,12)
plot(resLP(:,(12-1)*6+1:(12-1)*6+3),"Linewidth",width)
xlim([0 xLimit])
ylim([-yLimit yLimit])
grid on
grid(gca,'minor')
xlabel('time [s]')
ylabel('Innovation Size [m]')
title('Relative Position 6-2') 
%}
%% Test guassian-ness of residual-lowpass

faultIndex = ceil(faultTime);
%figure;plot(discLowPass)
%hisPlot(residualOutput(2000:end,:),faultIndex,2)
% Compute Covariance Matric once residual has stabilized

resMean = mean(residualOutput(2000:end,:));
%{
Qr = cov(residualOutput(2000:end,:));
save('highNoise_covariance_Mean_Empirical.mat','Qr','resMean')
%}
%% Fault Decision making

figure;
subplot(1,2,1)
grid on
grid(gca,'minor')
plot(detOutput)
ax = gca;
ax.FontSize = mediumSize; 

subplot(1,2,2)
grid on
grid(gca,'minor')
plot(isoOutput,'*')
ax = gca;
ax.FontSize = mediumSize; 

ylimit = max(fdiOutput(:));
linewidth = 1.5;
figure;
sgtitle("Thruster Fault Tests" )
subplot(2,3,1)
plot(fdiOutput(:,0*6 + 1),'linewidth',linewidth);
hold on
plot(fdiOutput(:,0*6 + 2),'linewidth',linewidth);
plot(fdiOutput(:,0*6 + 3),'linewidth',linewidth);
plot(fdiOutput(:,0*6 + 4),'linewidth',linewidth);
plot(fdiOutput(:,0*6 + 5),'linewidth',linewidth);
plot(fdiOutput(:,0*6 + 6),'linewidth',linewidth);
grid on
grid(gca,'minor')
title("Satellite 1 Tests" )
ylim([0,ylimit])
xlabel("Time [s]")
ylabel("CUSUM Test Value [-]")
ax = gca;
ax.FontSize = mediumSize; 

subplot(2,3,2)
line1 = plot(fdiOutput(:,1*6 + 1),'linewidth',linewidth);
hold on
line2 = plot(fdiOutput(:,1*6 + 2),'linewidth',linewidth);
line3 = plot(fdiOutput(:,1*6 + 3),'linewidth',linewidth);
line4 = plot(fdiOutput(:,1*6 + 4),'linewidth',linewidth);
line5 = plot(fdiOutput(:,1*6 + 5),'linewidth',linewidth);
line6 = plot(fdiOutput(:,1*6 + 6),'linewidth',linewidth);
grid on
grid(gca,'minor')
title("Satellite 2 Tests"  )
ylim([0,ylimit])
xlabel("Time [s]")
ylabel("CUSUM Test Value [-]")
ax = gca;
ax.FontSize = mediumSize; 

subplot(2,3,3)
plot(fdiOutput(:,2*6 + 1),'linewidth',linewidth);
hold on
plot(fdiOutput(:,2*6 + 2),'linewidth',linewidth);
plot(fdiOutput(:,2*6 + 3),'linewidth',linewidth);
plot(fdiOutput(:,2*6 + 4),'linewidth',linewidth);
plot(fdiOutput(:,2*6 + 5),'linewidth',linewidth);
plot(fdiOutput(:,2*6 + 6),'linewidth',linewidth);
grid on
grid(gca,'minor')
title("Satellite 3 Tests" )
ylim([0,ylimit])
xlabel("Time [s]")
ylabel("CUSUM Test Value [-]")
ax = gca;
ax.FontSize = mediumSize; 

subplot(2,3,4)
plot(fdiOutput(:,3*6 + 1),'linewidth',linewidth);
hold on
plot(fdiOutput(:,3*6 + 2),'linewidth',linewidth);
plot(fdiOutput(:,3*6 + 3),'linewidth',linewidth);
plot(fdiOutput(:,3*6 + 4),'linewidth',linewidth);
plot(fdiOutput(:,3*6 + 5),'linewidth',linewidth);
plot(fdiOutput(:,3*6 + 6),'linewidth',linewidth);
grid on
grid(gca,'minor')
title("Satellite 4 Tests" )
ylim([0,ylimit])
xlabel("Time [s]")
ylabel("CUSUM Test Value [-]")
ax = gca;
ax.FontSize = mediumSize; 

subplot(2,3,5)
plot(fdiOutput(:,4*6 + 1),'linewidth',linewidth);
hold on
plot(fdiOutput(:,4*6 + 2),'linewidth',linewidth);
plot(fdiOutput(:,4*6 + 3),'linewidth',linewidth);
plot(fdiOutput(:,4*6 + 4),'linewidth',linewidth);
plot(fdiOutput(:,4*6 + 5),'linewidth',linewidth);
plot(fdiOutput(:,4*6 + 6),'linewidth',linewidth);
grid on
grid(gca,'minor')
title("Satellite 5 Tests" )
ylim([0,ylimit])
xlabel("Time [s]")
ylabel("CUSUM Test Value [-]")
ax = gca;
ax.FontSize = mediumSize; 

subplot(2,3,6)
plot(fdiOutput(:,5*6 + 1),'linewidth',linewidth);
hold on
plot(fdiOutput(:,5*6 + 2),'linewidth',linewidth);
plot(fdiOutput(:,5*6 + 3),'linewidth',linewidth);
plot(fdiOutput(:,5*6 + 4),'linewidth',linewidth);
plot(fdiOutput(:,5*6 + 5),'linewidth',linewidth);
plot(fdiOutput(:,5*6 + 6),'linewidth',linewidth);
grid on
grid(gca,'minor')
title("Satellite 6 Tests" )
ylim([0,ylimit])
xlabel("Time [s]")
ylabel("CUSUM Test Value [-]")
ax = gca;
ax.FontSize = mediumSize; 

hL = legend([line1,line2,line3,line4,line5,line6],{'Thruster 1','Thruster 2','Thruster 3','Thruster 4','Thruster 5','Thruster 6'});
% Programatically move the Legend
newPosition = [0.85 0.6 0.2 0.2];
newUnits = 'normalized';
set(hL,'Position', newPosition,'Units', newUnits);

%}
if sFormation.spacecraftArray(1).faultDetected
    detectionTime = sFormation.spacecraftArray(1).faultDetTime;
    delay = detectionTime-faultTime
end
