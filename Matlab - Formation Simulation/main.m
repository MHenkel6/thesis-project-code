%clear all;
close all;
clc;
addpath('HelperFunctions')
addpath('DisturbanceForces')
addpath('Environment')
addpath('Environment/igrf')
addpath('Environment/igrf/datfiles')
addpath('FormationObjects')
%% Setup
mediumSize = 14;
% Initial value
Re = Constants.graviPar.Re/1000;
h = 687.4800;
inc =  62.8110/360*2*pi;
seed = 1371;
rng(seed); 
nSat = 6; 
type = 'octahedron';
center = [Re+h,0,0]*1000;
GM = Constants.muEarth;
velCenter = [0,sqrt(GM/(norm(center)))*cos(inc),sqrt(GM/(norm(center)))*sin(inc)]; % Reference velocity of center point
muEarth = GM;
T = 2*pi*sqrt(((Re+h)*1000)^3/muEarth);
dist = 142.3900;
formationOrientation = [0.9328,-0.0854,0.0314,0.3488];

% Control Parameters
controltype = 2;
disctype = 2;
thrustInterval = 10;
burnTime = 5;
thrust = 4;
navChange = 0;

faultType = 0; % faulttype, 1 is closed, 2 is open
faultTime = 1859; % random time after settle down (1000s for Kalman filter) in the oscillation
faultParam = 0.1;
satNo = 1 ;
thrusterNo = 1;
alpha = 1; % multiplicative error factor

% Spacecraft Parameters
mass = 100; 
spacecraftSize = [1,1,1];
inertia = [100,0,0;
           0,100,0;
           0,0,100];
isp = 200;
mib = 0.1;

selfBiasSize = alpha*1;
relBiasSize = alpha*0.05;
velRelBiasSize = alpha*0.0005;
selfNoise = alpha*2;
relNoise = alpha*0.02;
velRelNoise = alpha*0.0005;
posErrWeight = 1e2;
velErrWeight = 1e2;
if navChange
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
% Load covariance matrix and m         
% Initialize formation
sFormation = formation(nSat,type,center,velCenter,dist,formationOrientation,...
                       controltype,disctype,spacecraftParameters,0,0,...
                       navChange+1,navArray);
sFormation.disturbancesOn = 0;

sFormation.setFault(faultTime,satNo,thrusterNo,faultType,faultParam)

% Orbit time and time step
t0 = 0;
dt = Constants.dt;
nOrbit = 1;
tE = round(nOrbit*T/dt)*dt;

it= 1;
steps = round(tE/dt);
output = zeros((steps)*nSat,6);
relMeasOutput = zeros((steps)*4*nSat,6);
%% Integration
tic
for t = t0:dt:tE-dt
    % solve dynamics equation using rk4
    sFormation.rk4Prop(t,dt)
    formationState =  sFormation.getStates();
    output((it-1)*nSat+1:it*nSat,1:6) = formationState;
    relMeasOutput((it-1)*4*nSat+1:it*4*nSat,1:6) = sFormation.getRelStates(false);
    % Check if orbits explode
    if any(euclidnorm(formationState(:,1:3)) - 1000*Re > 5E6 )
        break
    end
    it = it+1;
    percent = (it-2)/steps*100;% Percentage indicator, not simulation relevant
    if mod(percent,5) < 1/steps*100
        disp(string(percent)+ ' %');
    end
end
toc
faultTime = sFormation.spacecraftArray(satNo).faultTime;
%% Post Processing
% Determine average thruster opening time
openingTimes = zeros(36,1);
openingCount = zeros(36,1);
for ii = 1:6
    sat = sFormation.spacecraftArray(ii);
    openingTimes(6*(ii-1)+1:6*(ii)) = sat.thrusterOpeningTime;
    openingCount(6*(ii-1)+1:6*(ii)) = sat.thrusterOpeningCount;
end
meanOpeningTime = mean(openingTimes./openingCount)
stdOpeningTime = std(openingTimes./openingCount)
posSat1 = output(1:6:end,1:3);
posSat2 = output(2:6:end,1:3);
posSat3 = output(3:6:end,1:3);
posSat4 = output(4:6:end,1:3);
posSat5 = output(5:6:end,1:3);
posSat6 = output(6:6:end,1:3);

h1 = euclidnorm(posSat1)-Re;
h2 = euclidnorm(posSat2)-Re;
h3 = euclidnorm(posSat3)-Re;
h4 = euclidnorm(posSat4)-Re;
h5 = euclidnorm(posSat5)-Re;
h6 = euclidnorm(posSat6)-Re;


% determine relative distances
vecdist12 = -(posSat1-posSat2);
vecdist13 = -(posSat1-posSat3);
vecdist14 = -(posSat1-posSat4);
vecdist15 = -(posSat1-posSat5);

vecdist23 = -(posSat2-posSat3);
vecdist34 = -(posSat3-posSat4);
vecdist45 = -(posSat4-posSat5);
vecdist52 = -(posSat5-posSat2);

vecdist62 = -(posSat6-posSat2);
vecdist63 = -(posSat6-posSat3);
vecdist64 = -(posSat6-posSat4);
vecdist65 = -(posSat6-posSat5);



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

distMeas12 = euclidnorm(relMeasOutput(1:24:end,1:3));
distMeas13 = euclidnorm(relMeasOutput(2:24:end,1:3));
distMeas14 = euclidnorm(relMeasOutput(3:24:end,1:3));
distMeas15 = euclidnorm(relMeasOutput(4:24:end,1:3));

distMeas21 = euclidnorm(relMeasOutput(5:24:end,1:3));
distMeas25 = euclidnorm(relMeasOutput(6:24:end,1:3));
distMeas26 = euclidnorm(relMeasOutput(7:24:end,1:3));
distMeas23 = euclidnorm(relMeasOutput(8:24:end,1:3));

distMeas31 = euclidnorm(relMeasOutput(9:24:end,1:3));
distMeas32 = euclidnorm(relMeasOutput(10:24:end,1:3));
distMeas36 = euclidnorm(relMeasOutput(11:24:end,1:3));
distMeas34 = euclidnorm(relMeasOutput(12:24:end,1:3));

distMeas43 = euclidnorm(relMeasOutput(13:24:end,1:3));
distMeas46 = euclidnorm(relMeasOutput(14:24:end,1:3));
distMeas45 = euclidnorm(relMeasOutput(15:24:end,1:3));
distMeas41 = euclidnorm(relMeasOutput(16:24:end,1:3));

distMeas54 = euclidnorm(relMeasOutput(17:24:end,1:3));
distMeas56 = euclidnorm(relMeasOutput(18:24:end,1:3));
distMeas52 = euclidnorm(relMeasOutput(19:24:end,1:3));
distMeas51 = euclidnorm(relMeasOutput(20:24:end,1:3));

distMeas65 = euclidnorm(relMeasOutput(21:24:end,1:3));
distMeas64 = euclidnorm(relMeasOutput(22:24:end,1:3));
distMeas63 = euclidnorm(relMeasOutput(23:24:end,1:3));
distMeas62 = euclidnorm(relMeasOutput(24:24:end,1:3));

vecMeas12 = relMeasOutput(1:24:end,1:3);
vecMeas13 = relMeasOutput(2:24:end,1:3);
vecMeas14 = relMeasOutput(3:24:end,1:3);
vecMeas15 = relMeasOutput(4:24:end,1:3);

vecMeas21 = relMeasOutput(5:24:end,1:3);
vecMeas25 = relMeasOutput(6:24:end,1:3);
vecMeas26 = relMeasOutput(7:24:end,1:3);
vecMeas23 = relMeasOutput(8:24:end,1:3);

vecMeas31 = relMeasOutput(9:24:end,1:3);
vecMeas32 = relMeasOutput(10:24:end,1:3);
vecMeas36 = relMeasOutput(11:24:end,1:3);
vecMeas34 = relMeasOutput(12:24:end,1:3);

vecMeas43 = relMeasOutput(13:24:end,1:3);
vecMeas46 = relMeasOutput(14:24:end,1:3);
vecMeas45 = relMeasOutput(15:24:end,1:3);
vecMeas41 = relMeasOutput(16:24:end,1:3);

vecMeas54 = relMeasOutput(17:24:end,1:3);
vecMeas56 = relMeasOutput(18:24:end,1:3);
vecMeas52 = relMeasOutput(19:24:end,1:3);
vecMeas51 = relMeasOutput(20:24:end,1:3);

vecMeas65 = relMeasOutput(21:24:end,1:3);
vecMeas64 = relMeasOutput(22:24:end,1:3);
vecMeas63 = relMeasOutput(23:24:end,1:3);
vecMeas62 = relMeasOutput(24:24:end,1:3);

DVarray = zeros(6,1);
for ii = 1:6
    DVarray(ii) = sFormation.spacecraftArray(ii).accumDV;
end
DVarray
mean(DVarray)
%sFormation.spacecraftArray.thrusterOpeningCount
%sFormation.spacecraftArray.thrusterOpeningTime
%sFormation.spacecraftArray.spentProp
% Determine location in Formation Frame
FFloc = (posSat1 + posSat2 + posSat3 + posSat4 + posSat5 + posSat6)/6; %location of Formation Frame
posSat1FF = quatrotate(quatinv(formationOrientation),posSat1-FFloc);
posSat2FF = quatrotate(quatinv(formationOrientation),posSat2-FFloc);
posSat3FF = quatrotate(quatinv(formationOrientation),posSat3-FFloc);
posSat4FF = quatrotate(quatinv(formationOrientation),posSat4-FFloc);
posSat5FF = quatrotate(quatinv(formationOrientation),posSat5-FFloc);
posSat6FF = quatrotate(quatinv(formationOrientation),posSat6-FFloc);
%% Plots

%Orbit plot
%Earth sphere

figure
earth_sphere(100,'m')
hold on;
plot3(posSat1(:,1),posSat1(:,2),posSat1(:,3))
plot3(posSat2(:,1),posSat2(:,2),posSat2(:,3))
plot3(posSat3(:,1),posSat3(:,2),posSat3(:,3))
plot3(posSat4(:,1),posSat4(:,2),posSat4(:,3))
plot3(posSat5(:,1),posSat5(:,2),posSat5(:,3))
plot3(posSat6(:,1),posSat6(:,2),posSat6(:,3))

xlabel('X [km]')
ylabel('Y [km]')
zlabel('Z [km]')
xlim([posSat1(1,1)-1000,posSat1(1,1)+1000])
ylim([posSat1(1,2)-1000,posSat1(1,2)+1000])
zlim([posSat1(1,3)-1000,posSat1(1,3)+1000])
%% Altitude Plots
time = t0:dt:tE-dt;
figure;
plot(time,h1);
hold on
plot(time,h2);
plot(time,h3);
plot(time,h4);
plot(time,h5);
plot(time,h6);
xlabel('Time [s]')
ylabel('Altitudes of Satellites [m]')
%% Measurement Error plot
errVec12 = vecMeas12-vecdist12;
errVec13 = vecMeas13-vecdist13;
errVec14 = vecMeas14-vecdist14;
errVec15 = vecMeas15-vecdist15;

errVec21 = vecMeas21-(-1*vecdist12);
errVec25 = vecMeas25-(-1*vecdist52);
errVec26 = vecMeas26-(-1*vecdist62);
errVec23 = vecMeas23-vecdist23;

errVec31 = vecMeas31-(-1*vecdist13);
errVec32 = vecMeas32-(-1*vecdist23);
errVec36 = vecMeas36-(-1*vecdist63);
errVec34 = vecMeas34-vecdist34;

errVec43 = vecMeas43-(-1*vecdist34);
errVec46 = vecMeas46-(-1*vecdist64);
errVec45 = vecMeas45-vecdist45;
errVec41 = vecMeas41-(-1*vecdist14);

errVec54 = vecMeas54-(-1*vecdist45);
errVec56 = vecMeas56-(-1*vecdist65);
errVec52 = vecMeas52-vecdist52;
errVec51 = vecMeas51-(-1*vecdist15);

errVec65 = vecMeas65-vecdist65;
errVec64 = vecMeas64-vecdist64;
errVec63 = vecMeas63-vecdist63;
errVec62 = vecMeas62-vecdist62;
errs = [errVec12,errVec13,errVec14,errVec15,...
        errVec21,errVec25,errVec26,errVec23,...
        errVec31,errVec32,errVec36,errVec34,...
        errVec43,errVec46,errVec45,errVec41,...
        errVec54,errVec56,errVec52,errVec51,...
        errVec65,errVec64,errVec63,errVec62];

figure;
plot(errVec12)
hold on
plot(errVec13)
plot(errVec14)
plot(errVec15)
rmsComp = rms(errs);
rmsPlot = zeros(1,6);
for ii = 1:6
    rmsPlot(ii) = mean(rmsComp(6*(ii-1)+1:6*ii));
end
figure

bar(rmsPlot)
%% Relative Positioning Plot to sat 1
s1 = sFormation.spacecraftArray(1);
s2 = sFormation.spacecraftArray(2);
s3 = sFormation.spacecraftArray(3);
s4 = sFormation.spacecraftArray(4);
s5 = sFormation.spacecraftArray(5);
s6 = sFormation.spacecraftArray(6);
tSettle = 1;
time = tSettle-1:dt:tE-dt;
ylimits = [-2,3];

figure
t = tiledlayout(3,4,'TileSpacing','Compact','Padding','Compact');
nexttile
plot(time,vecdist12(tSettle:end,1)-s1.n1Offset(1),'r')
hold on
plot(time,vecdist12(tSettle:end,2)-s1.n1Offset(2),'g')
plot(time,vecdist12(tSettle:end,3)-s1.n1Offset(3),'b')
plot([faultTime,faultTime],ylimits,'k')
ylim(ylimits)
title("Satellite Link 1-2")
grid on;
grid minor;
xlabel('Time [s]')
ylabel('Relative Distance [m]')
legend("X distance", "Y distance", "Z distance","Location", "northeast")

nexttile
plot(time,vecdist13(tSettle:end,1)-s1.n2Offset(1),'r')
hold on
plot(time,vecdist13(tSettle:end,2)-s1.n2Offset(2),'g')
plot(time,vecdist13(tSettle:end,3)-s1.n2Offset(3),'b')
plot([faultTime,faultTime],ylimits,'k')
ylim(ylimits)
title("Satellite Link 1-3")
grid on;
grid minor;
xlabel('Time [s]')
ylabel('Relative Distance [m]')
legend("X distance", "Y distance", "Z distance","Location", "northeast")

nexttile
plot(time,vecdist14(tSettle:end,1)-s1.n3Offset(1),'r')
hold on
plot(time,vecdist14(tSettle:end,2)-s1.n3Offset(2),'g')
plot(time,vecdist14(tSettle:end,3)-s1.n3Offset(3),'b')
plot([faultTime,faultTime],ylimits,'k')
ylim(ylimits)
title("Satellite Link 1-4")
grid on;
grid minor;
xlabel('Time [s]')
ylabel('Relative Distance [m]')
legend("X distance", "Y distance", "Z distance","Location", "northeast")

nexttile
plot(time,vecdist15(tSettle:end,1)-s1.n4Offset(1),'r')
hold on
plot(time,vecdist15(tSettle:end,2)-s1.n4Offset(2),'g')
plot(time,vecdist15(tSettle:end,3)-s1.n4Offset(3),'b')
plot([faultTime,faultTime],ylimits,'k')
ylim(ylimits)
title("Satellite Link 1-5")
grid on;
grid minor;
xlabel('Time [s]')
ylabel('Relative Distance[m]')
legend("X distance", "Y distance", "Z distance","Location", "northeast")

nexttile
plot(time,vecdist23(tSettle:end,1)-s2.n4Offset(1),'r')
hold on
plot(time,vecdist23(tSettle:end,2)-s2.n4Offset(2),'g')
plot(time,vecdist23(tSettle:end,3)-s2.n4Offset(3),'b')
plot([faultTime,faultTime],ylimits,'k')
ylim(ylimits)
title("Satellite Link 2-3")
grid on;
grid minor;
xlabel('Time [s]')
ylabel('Relative Distance [m]')
legend("X distance", "Y distance", "Z distance","Location", "northeast")

nexttile
plot(time,vecdist34(tSettle:end,1)-s3.n4Offset(1),'r')
hold on
plot(time,vecdist34(tSettle:end,2)-s3.n4Offset(2),'g')
plot(time,vecdist34(tSettle:end,3)-s3.n4Offset(3),'b')
plot([faultTime,faultTime],ylimits,'k')
ylim(ylimits)
title("Satellite Link 3-4")
grid on;
grid minor;
xlabel('Time [s]')
ylabel('Relative Distance [m]')
legend("X distance", "Y distance", "Z distance","Location", "northeast")

nexttile
plot(time,vecdist45(tSettle:end,1)-s4.n3Offset(1),'r')
hold on
plot(time,vecdist45(tSettle:end,2)-s4.n3Offset(2),'g')
plot(time,vecdist45(tSettle:end,3)-s4.n3Offset(3),'b')
plot([faultTime,faultTime],ylimits,'k')
ylim(ylimits)
title("Satellite Link 4-5")
grid on;
grid minor;
xlabel('Time [s]')
ylabel('Relative Distance [m]')
legend("X distance", "Y distance", "Z distance","Location", "northeast")

nexttile
plot(time,vecdist52(tSettle:end,1)-s5.n3Offset(1),'r')
hold on
plot(time,vecdist52(tSettle:end,2)-s5.n3Offset(2),'g')
plot(time,vecdist52(tSettle:end,3)-s5.n3Offset(3),'b')
plot([faultTime,faultTime],ylimits,'k')
ylim(ylimits)
title("Satellite Link 5-2")
grid on;
grid minor;
xlabel('Time [s]')
ylabel('Relative Distance [m]')
legend("X distance", "Y distance", "Z distance","Location", "northeast")

nexttile
plot(time,vecdist62(tSettle:end,1)-s6.n4Offset(1),'r')
hold on
plot(time,vecdist62(tSettle:end,2)-s6.n4Offset(2),'g')
plot(time,vecdist62(tSettle:end,3)-s6.n4Offset(3),'b')
plot([faultTime,faultTime],ylimits,'k')
ylim(ylimits)
title("Satellite Link 6-2")
grid on;
grid minor;
xlabel('Time [s]')
ylabel('Relative Distance [m]')
legend("X distance", "Y distance", "Z distance","Location", "northeast")

nexttile
plot(time,vecdist63(tSettle:end,1)-s6.n3Offset(1),'r')
hold on
plot(time,vecdist63(tSettle:end,2)-s6.n3Offset(2),'g')
plot(time,vecdist63(tSettle:end,3)-s6.n3Offset(3),'b')
plot([faultTime,faultTime],ylimits,'k')
ylim(ylimits)
title("Satellite Link 6-3")
grid on;
grid minor;
xlabel('Time [s]')
ylabel('Relative Distance [m]')
legend("X distance", "Y distance", "Z distance","Location", "northeast")

nexttile
plot(time,vecdist64(tSettle:end,1)-s6.n2Offset(1),'r')
hold on
plot(time,vecdist64(tSettle:end,2)-s6.n2Offset(2),'g')
plot(time,vecdist64(tSettle:end,3)-s6.n2Offset(3),'b')
plot([faultTime,faultTime],ylimits,'k')
ylim(ylimits)
title("Satellite Link 6-4")
grid on;
grid minor;
xlabel('Time [s]')
ylabel('Relative Distance [m]')
legend("X distance", "Y distance", "Z distance","Location", "northeast")

nexttile
plot(time,vecdist65(tSettle:end,1)-s6.n1Offset(1),'r')
hold on
plot(time,vecdist65(tSettle:end,2)-s6.n1Offset(2),'g')
plot(time,vecdist65(tSettle:end,3)-s6.n1Offset(3),'b')
plot([faultTime,faultTime],ylimits,'k')
ylim(ylimits)
title("Satellite Link 6-5")
grid on;
grid minor;
xlabel('Time [s]')
ylabel('Relative Distance [m]')
legend("X distance", "Y distance", "Z distance","Location", "northeast")


%% Actual Relative Distance plot
tSettle = 1;
time = tSettle-1:dt:tE-dt;
figure('units','normalized','outerposition',[0 0.2 1 0.55]);
yLimMax = max([dist12(50:end);dist13(50:end);dist14(50:end);dist15(50:end)])*(1+0.025);
yLimMin = min([dist12(50:end);dist13(50:end);dist14(50:end);dist15(50:end)])*(1-0.005);

% Satellite 1
subplot(2,3,1)
plot(time, dist12(tSettle:end),'k')
hold on;
plot(time, dist13(tSettle:end),'r')
plot(time, dist14(tSettle:end),'g')
plot(time, dist15(tSettle:end),'b')
plot([time(1),time(end)],[dist,dist]);
ylim([yLimMin,yLimMax])
xlabel('Time [s]')
ylabel('Relative Distance [m]')
title("Satellite 1 Connections")

grid on
grid minor 
ax = gca;
ax.FontSize = mediumSize; 

% Satellite 2
subplot(2,3,2)
plot(time, dist21(tSettle:end),'k')
hold on;
plot(time, dist25(tSettle:end),'r')
plot(time, dist26(tSettle:end),'g')
plot(time, dist23(tSettle:end),'b')
plot([time(1),time(end)],[dist,dist]);
ylim([yLimMin,yLimMax])
xlabel('Time [s]')
ylabel('Relative Distance [m]')
title("Satellite 2 Connections")
grid on
grid minor 
ax = gca;
ax.FontSize = mediumSize; 

% Satellite 3
subplot(2,3,3)
plot(time, dist31(tSettle:end),'k')
hold on;
plot(time, dist32(tSettle:end),'r')
plot(time, dist36(tSettle:end),'g')
plot(time, dist34(tSettle:end),'b')
plot([time(1),time(end)],[dist,dist]);
ylim([yLimMin,yLimMax])
xlabel('Time [s]')
ylabel('Relative Distance [m]')
title("Satellite 3 Connections")
grid on
grid minor 
ax = gca;
ax.FontSize = mediumSize; 

legend('Distance X-1','Distance X-2','Distance X-3','Distance X-4','Nominal Distance','Location','NorthEast')
% Satellite 4
subplot(2,3,4)
plot(time, dist43(tSettle:end),'k')
hold on;
plot(time, dist46(tSettle:end),'r')
plot(time, dist45(tSettle:end),'g')
plot(time, dist41(tSettle:end),'b')

plot([time(1),time(end)],[dist,dist]);
ylim([yLimMin,yLimMax])
xlabel('Time [s]')
ylabel('Relative Distance [m]')
title("Satellite 4 Connections")
grid on
grid minor 
ax = gca;
ax.FontSize = mediumSize; 

% Satellite 5
subplot(2,3,5)
plot(time, dist54(tSettle:end),'k')
hold on;
plot(time, dist56(tSettle:end),'r')
plot(time, dist52(tSettle:end),'g')
plot(time, dist51(tSettle:end),'b')
plot([time(1),time(end)],[dist,dist]);
ylim([yLimMin,yLimMax])
xlabel('Time [s]')
ylabel('Relative Distance [m]')
title("Satellite 5 Connections")
grid on
grid minor 
ax = gca;
ax.FontSize = mediumSize; 

% Satellite 6
subplot(2,3,6)
plot(time, dist65(tSettle:end),'k')
hold on;
plot(time, dist64(tSettle:end),'r')
plot(time, dist63(tSettle:end),'g')
plot(time, dist62(tSettle:end),'b')
plot([time(1),time(end)],[dist,dist]);
ylim([yLimMin,yLimMax])
xlabel('Time [s]')
ylabel('Relative Distance [m]')
grid on
grid minor 
title("Satellite 6 Connections")
ax = gca;
ax.FontSize = mediumSize; 
%{
tSettle = 1;
time = tSettle-1:dt:tE-dt;
figure;

% Satellite 1
subplot(2,3,1)
plot(time, dist12(tSettle:end),'k')
hold on;
plot(time, dist13(tSettle:end),'r')
plot(time, dist14(tSettle:end),'g')
plot(time, dist15(tSettle:end),'b')
plot([time(1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance [m]')
title("Satellite 1 Connections")

grid on
grid minor 
ax = gca;
ax.FontSize = mediumSize; 

% Satellite 2
subplot(2,3,2)
plot(time, dist21(tSettle:end),'k')
hold on;
plot(time, dist25(tSettle:end),'r')
plot(time, dist26(tSettle:end),'g')
plot(time, dist23(tSettle:end),'b')
plot([time(1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance [m]')
title("Satellite 2 Connections")
grid on
grid minor 
ax = gca;
ax.FontSize = mediumSize; 

% Satellite 3
subplot(2,3,3)
plot(time, dist31(tSettle:end),'k')
hold on;
plot(time, dist32(tSettle:end),'r')
plot(time, dist36(tSettle:end),'g')
plot(time, dist34(tSettle:end),'b')
plot([time(1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance [m]')
title("Satellite 3 Connections")
grid on
grid minor 
ax = gca;
ax.FontSize = mediumSize; 

% Satellite 4
subplot(2,3,4)
plot(time, dist43(tSettle:end),'k')
hold on;
plot(time, dist46(tSettle:end),'r')
plot(time, dist45(tSettle:end),'g')
plot(time, dist41(tSettle:end),'b')

plot([time(1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 4 [m]')
grid on
grid minor 
ax = gca;
ax.FontSize = mediumSize; 

% Satellite 5
subplot(2,3,5)
plot(time, dist54(tSettle:end),'k')
hold on;
plot(time, dist56(tSettle:end),'r')
plot(time, dist52(tSettle:end),'g')
plot(time, dist51(tSettle:end),'b')
plot([time(1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 5 [m]')
grid on
grid minor 
ax = gca;
ax.FontSize = mediumSize; 

% Satellite 6
subplot(2,3,6)
plot(time, dist65(tSettle:end),'k')
hold on;
plot(time, dist64(tSettle:end),'r')
plot(time, dist63(tSettle:end),'g')
plot(time, dist62(tSettle:end),'b')
plot([time(1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 6 [m]')
grid on
grid minor 
ax = gca;
ax.FontSize = mediumSize; 

legend('Distance X-1','Distance X-2','Distance X-3','Distance X-4','Nominal Distance','Location','NorthEast')
%}
%%  Measured Relative Distance plot
tSettle = 1;
time = tSettle-1:dt:tE-dt;
figure;

% Satellite 1
subplot(2,3,1)
plot(time, distMeas12(tSettle:end),'k')
hold on;
plot(time, distMeas13(tSettle:end),'r')
plot(time, distMeas14(tSettle:end),'g')
plot(time, distMeas15(tSettle:end),'b')
%plot(time, dist16)
plot([time(1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 1 [m]')

% Satellite 2
subplot(2,3,2)
plot(time, distMeas21(tSettle:end),'k')
hold on;
plot(time, distMeas25(tSettle:end),'r')
plot(time, distMeas26(tSettle:end),'g')
plot(time, distMeas23(tSettle:end),'b')
%plot(time, dist16)
plot([time(1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 2 [m]')

% Satellite 3
subplot(2,3,3)
plot(time, distMeas31(tSettle:end),'k')
hold on;
plot(time, distMeas32(tSettle:end),'r')
plot(time, distMeas36(tSettle:end),'g')
plot(time, distMeas34(tSettle:end),'b')

%plot(time, dist16)
plot([time(1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 3 [m]')

% Satellite 4
subplot(2,3,4)
plot(time, distMeas43(tSettle:end),'k')
hold on;
plot(time, distMeas46(tSettle:end),'r')
plot(time, distMeas45(tSettle:end),'g')
plot(time, distMeas41(tSettle:end),'b')

plot([time(1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 4 [m]')

% Satellite 5
subplot(2,3,5)
plot(time, distMeas54(tSettle:end),'k')
hold on;
plot(time, distMeas56(tSettle:end),'r')
plot(time, distMeas52(tSettle:end),'g')
plot(time, distMeas51(tSettle:end),'b')

%plot(time, dist16)
plot([time(1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 5 [m]')

% Satellite 6
subplot(2,3,6)
plot(time, distMeas65(tSettle:end),'k')
hold on;
plot(time, distMeas64(tSettle:end),'r')
plot(time, distMeas63(tSettle:end),'g')
plot(time, distMeas62(tSettle:end),'b')

%plot(time, dist16)
plot([time(1),time(end)],[dist,dist]);
xlabel('Time [s]')
ylabel('Relative Distance to Satellite 6 [m]')

%% Relative Distance plots
%{
% Distance to sat 2
figure
plot(time, vecdist12(:,1))
hold on
plot(time, vecdist12(:,2))
plot(time, vecdist12(:,3))
% Distance to sat 4
figure
plot(time, vecdist14(:,1))
hold on
plot(time, vecdist14(:,2))
plot(time, vecdist14(:,3))
%}
%% Relative Motion in Formation Frame Before Fault
faultIndex = min(floor(faultTime),5902);
linewidth = 6;
figure
plot3(posSat1FF(tSettle:faultIndex,1),posSat1FF(tSettle:faultIndex,2),posSat1FF(tSettle:faultIndex,3),'LineWidth',linewidth)
hold on;
plot3(posSat2FF(tSettle:faultIndex,1),posSat2FF(tSettle:faultIndex,2),posSat2FF(tSettle:faultIndex,3),'LineWidth',linewidth)
plot3(posSat3FF(tSettle:faultIndex,1),posSat3FF(tSettle:faultIndex,2),posSat3FF(tSettle:faultIndex,3),'LineWidth',linewidth)
plot3(posSat4FF(tSettle:faultIndex,1),posSat4FF(tSettle:faultIndex,2),posSat4FF(tSettle:faultIndex,3),'LineWidth',linewidth)
plot3(posSat5FF(tSettle:faultIndex,1),posSat5FF(tSettle:faultIndex,2),posSat5FF(tSettle:faultIndex,3),'LineWidth',linewidth)
plot3(posSat6FF(tSettle:faultIndex,1),posSat6FF(tSettle:faultIndex,2),posSat6FF(tSettle:faultIndex,3),'LineWidth',linewidth)
grid on;
xlabel('X [km]')
ylabel('Y [km]')
zlabel('Z [km]')
legend('Satellite 1','Satellite 2','Satellite 3','Satellite 4','Satellite 5','Satellite 6')
%% Relative Motion in Formation Frame After Fault
linewidth = 6;
figure
plot3(posSat1FF(faultIndex:end,1),posSat1FF(faultIndex:end,2),posSat1FF(faultIndex:end,3),'LineWidth',linewidth)
hold on;
plot3(posSat2FF(faultIndex:end,1),posSat2FF(faultIndex:end,2),posSat2FF(faultIndex:end,3),'LineWidth',linewidth)
plot3(posSat3FF(faultIndex:end,1),posSat3FF(faultIndex:end,2),posSat3FF(faultIndex:end,3),'LineWidth',linewidth)
plot3(posSat4FF(faultIndex:end,1),posSat4FF(faultIndex:end,2),posSat4FF(faultIndex:end,3),'LineWidth',linewidth)
plot3(posSat5FF(faultIndex:end,1),posSat5FF(faultIndex:end,2),posSat5FF(faultIndex:end,3),'LineWidth',linewidth)
plot3(posSat6FF(faultIndex:end,1),posSat6FF(faultIndex:end,2),posSat6FF(faultIndex:end,3),'LineWidth',linewidth)
grid on;
xlabel('X [km]')
ylabel('Y [km]')
zlabel('Z [km]')
legend('Satellite 1','Satellite 2','Satellite 3','Satellite 4','Satellite 5','Satellite 6')