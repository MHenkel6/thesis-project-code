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
seed = 1371; % random seed 
rng(seed); 
%% Script to verify orbit verification
% NOTE: The verification assumes no J2 effect, i.e. Keplerian orbits
%       Therefore the property 'maxDegree' in the 'Constants.m'file 
%       has to be set to 1!

% Initial values
testEllipse = false; % Bool to test elliptic, inclined case
Re = Constants.graviPar.Re/1000; % Earth radius
GM = Constants.muEarth; % Earth standard gravitational parameter

h = 1200; % orbital altitude
if testEllipse
    inc =  60/360*2*pi; % Inclination
    velMag = sqrt(3*GM/(2*(norm(pos)))); % Ellipse e = 0.5
else
    inc = 0;
    velMag = sqrt(GM/(norm(pos))); % Circular Orbit
end

pos = [Re+h,0,0]*1000;
vel = [0,velMag*cos(inc),velMag*sin(inc)]; % Reference velocity of center point

a = -GM/(velMag^2-2*GM/(norm(pos)));% (Re+h)*1000;
T = 2*pi*sqrt(a^3/GM);

% Orbit time and time step
t0 = 0;
dt = 0.5;
tE = 1.1*T;
state = [pos,vel];
it= 2;
steps = round(tE/dt);
output = zeros((steps),6);
output(1,1:6) = state;

%% Integration
for t = t0:dt:tE-dt
    % solve dynamics equation using rk4
    % Runge-Kutta 4 Integration Scheme
    pos = state(1:3);
    vel = state(4:6);
    dev1 = dynamics(t     , pos, vel,100,0,0);
    dev2 = dynamics(t+dt/2, pos + dt/2*dev1(1:3),vel + dt/2*dev1(4:6),...
                    100,0,0);
    dev3 = dynamics(t+dt/2, pos + dt/2*dev2(1:3),vel + dt/2*dev2(4:6),...
                    100,0,0);
    dev4 = dynamics(t+dt  , pos + dt*dev3(1:3),  vel + dt*dev3(4:6),...
                    100,0,0);                   
    devTotal = dt*(dev1 + 2*dev2 + 2*dev3 + dev4)/6;
    state = state + devTotal;
    output(it,1:6) = state;
    
    it = it+1;
    percent = (it-2)/steps*100;% Percentage indicator, not simulation relevant
    if mod(percent,5) < 1/steps*100
        disp(string(percent)+ ' %');
    end
end

%% Post Processing
posSat1 = output(1:end,1:3);
rperi = min(euclidnorm(posSat1))
rapo = max(euclidnorm(posSat1))
h1 = euclidnorm(posSat1)-Re;
dist = euclidnorm(posSat1(2:end,:)-posSat1(1,:));
[~,Tind] = min(dist);
Tsim = Tind*dt;
Tdiff = Tsim-T
if abs(Tdiff) < dt/2
    disp("Orbit verification test: passed")
end
% Plots
figure
earth_sphere(100,'m')
hold on
plot3(posSat1(:,1),posSat1(:,2),posSat1(:,3),'linewidth', 1)