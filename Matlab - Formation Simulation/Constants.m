%% Constants
classdef Constants
    properties( Constant = true )
        
        % Time constants
        startDate = juliandate(2019,9,17);
        dt = 1;
        % Graviational parameters
        graviPar  = load('aeroegm2008.mat','GM','Re','degree','C','S'); % Pre load gravity module for speed;
        J2 = 0.0010826357;
        maxDegree = 2;
        Sun = load('sunPosition.mat','pos','tArray','deltaT');
        Moon = load('moonPosition.mat','pos','tArray','deltaT');
        igrfcoef =load('igrfcoefs.mat');
        muSun = 1.32712440018E20; % sun gravitational parameter
        muMoon = 4.9048695E12; % moon gravitational parameter
        muEarth = 3.986004418E14;
        planetRot = 2*pi/(23.9345*3600); % Earth rotation rate rad/s;
        spacecraftCharge = 1.9271e-08; % acquired electric charge [C]
        relBiasSize = 0.05;
        velRelBiasSize =  0.0005;
        selfNoise = 2;
        posRelNoise = 0.02;
        velRelNoise = 0.0005;
    end
end