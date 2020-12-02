clear all;
close all;
clc;
addpath('..')
addpath('../HelperFunctions')
addpath('../DisturbanceForces')
addpath('../Environment')
addpath('../Environment/igrf')
addpath('../Environment/igrf/datfiles')
addpath('../FormationObjects')
%% Script to verify the frame transformations

%% Quaternion Verification
axis = [0,1,0];
angle = 90*pi/180;
quattest = [cos(angle/2),
            axis(1)*sin(angle/2),
            axis(2)*sin(angle/2),
            axis(3)*sin(angle/2)];
p1 = [1,0,0];
p2 = [0,1,0];
p3 = [0,0,1];
p4 = [3,4,5];
prot1 = quatrotate(quattest',p1)
prot2 = quatrotate(quattest',p2)
prot3 = quatrotate(quattest',p3)
prot4 = quatrotate(quattest',p4)

seed = 6;
rng(seed)
qAtt = rand(4,1);
qAtt = qAtt/norm(qAtt);
testVector = [5,4,3];
f = quatrotate(qAtt',testVector);
finv = quatrotate(quatinv(qAtt'),f);
if euclidnorm(testVector-finv) < 1e-10
    disp("Quaternion Rotation verififed")
end

%% Hill to ECI frame Verification

h0 = 400;
Re = Constants.graviPar.Re; 
GM = Constants.graviPar.GM;
r0 = Re+h0;
inc = 60*pi/180;
pos0 = [0,r0,0];
velMag0 = sqrt(GM/norm(pos0));
vel0 = [-velMag0*cos(inc),0,velMag0*sin(inc)];
T0 = 2*pi*sqrt(norm(pos0)^3/GM);
n = 2*pi/T0 ;

pos1= [0,r0+1,0];
velMag1 = sqrt(GM/norm(pos1));
vel1 = [-velMag1*cos(inc),0,velMag1*sin(inc)];

xcomp = [1;0;0];
vcomp = [0; sqrt(GM/(norm(pos0)+1))- sqrt(GM/norm(pos0)) - n; 0];

% Construct Hill Frame
x = pos0/norm(pos0);
z = cross(x,vel0/norm(vel0));
y = cross(z,x)/(norm(cross(z,x)));
THillECI = [x;
            y;
            z];
        
argLat = 0 ;
O = pi/2;
THillECIAnalytical = rotZ(argLat)*rotX(inc)*rotZ(O);
TDiff = THillECIAnalytical-THillECI;

SHill = n*skewSym([0,0,1]);
SECI = n*skewSym(z);
TdotAnalytical = -SHill*THillECI;
Tdot = n*rotZdot(argLat)*rotX(inc)*rotZ(O);
TdotDiff = TdotAnalytical-Tdot;


xrel = THillECI*(pos1-pos0)';
vrel = THillECI*(vel1-vel0)' + Tdot * (pos1-pos0)';
if (euclidnorm(xcomp-xrel)<1e-10) & (euclidnorm(vcomp-vrel)<1e-10)
    if (norm(TDiff) <1e-10) & (norm(TdotDiff)<1e-10)
        disp("ECI to Hill Frame verified")
    end
end
%% ECI to ECEF Verification
p = [Re+1000*1000,0,0];

dt = 5; % use dt = 0.1 for more accurate results
TEarth = 23.9345;
timeArray = 0:dt:24*3600;
pList = zeros(3,length(timeArray));
for ii = 1:length(timeArray)
    time = timeArray(ii);
    rotECEF = rotZ(Constants.planetRot*time);
    pList(:,ii) = rotECEF*p';
end
dists = euclidnorm((pList(:,2:end)-pList(:,1))');
[valmin,indmin] = min(dists);
tmin = indmin*dt;
figure
earth_sphere(100,'m')
hold on
plot3(pList(1,1:18*3600/dt),pList(2,1:18*3600/dt),pList(3,1:18*3600/dt))
xlabel("X [m]")
ylabel("Y [m]")
zlabel("Time [s]")
grid on
grid minor
title("Movement in the ECEF Frame of a fixed point in the ECI frame") 
if abs(TEarth*3600-tmin) < dt/2
    disp("ECI to ECEF verified")
end
    
