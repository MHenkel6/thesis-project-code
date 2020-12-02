% This script creates the position array of the sun position w.r.t. the
% Earth for the calculation of solar radiation pressure and solar wind

%% Setup
clc;
close all;
clear all;

%% Get sun position
startDate = Constants.startDate; % Start date in julian calendar
tEnd = 5*24*3600;% End of simulation time,in s
deltaT = 5; % step size in s
arraySize = tEnd/deltaT;
pos = zeros(arraySize+1,3);
for i = 1:arraySize+1
    t = (i-1)*deltaT;
    pos(i,:) = 1000*planetEphemeris(startDate+t/86400, 'Moon','Earth');
end
tArray = 0:deltaT:tEnd;

%% Save data file
save('centerPosition.mat','pos','tArray','deltaT');


