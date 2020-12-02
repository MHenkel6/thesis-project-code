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
seed = 1378;
rng(seed,'combRecursive')
nStart = 1;
nEnd = 100;
fileName = 'DataKalman/CentralizedKalmanFilterEval_';


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
selfBiasSize = 1;
relBiasSize = 0.05;
velRelBiasSize =  0.0005;
selfNoise = 2;
relNoise = 0.02;
velRelNoise = 0.0005;
posErrWeight = 1e2; 
velErrWeight = 1e2;

load('covariance_Mean_Empirical.mat','Qr','resMean')

%% Integration
parfor ii = nStart:nEnd
    stream = RandStream.getGlobalStream();
    stream.Substream = ii;
    for faultType = 1:2 % faulttype, 1 is closed, 2 is open
        for satNo = 1:6
            for thrusterNo = 1:6
                spacecraftParameters = {mass, spacecraftSize, inertia, ...
                              thrustInterval,burnTime,thrust,isp,mib...
                              selfBiasSize, relBiasSize, velRelBiasSize...
                              selfNoise, relNoise,velRelNoise,...
                              posErrWeight,velErrWeight};
                % Load covariance matrix and m

                % Initialize formation
                sFormation = formation(nSat,type,center,velCenter,dist,formationOrientation,...
                                       controltype,disctype,spacecraftParameters,Qr,resMean);
                sFormation.disturbancesOn = 0;
                faultTime = rand*(0.5*T)+1000; % random time after settle down (300s) in the oscillation
                faultParam = 0.1+0.9*((ii-1)/(100-1));
                sFormation.setFault(faultTime,satNo,thrusterNo,faultType,faultParam)
                % Use linear or extended Kalman Filter



                % Orbit time and time step
                t0 = 0;
                dt = Constants.dt;
                nOrbit = 1;
                tE = round(nOrbit*T/dt)*dt;
                zeroed = false;
                it= 1;
                steps = round(tE/dt);
                fdiOutput = zeros(steps,6*nSat);
                detOutput = zeros(steps,1);
                isoOutput = zeros(steps,1);
                for t = t0:dt:tE-dt
                    % solve dynamics equation using rk4
                    sFormation.rk4Prop(t,dt)
                    [formationState,formationStateHill] =  sFormation.getRelStates(zeroed);

                    fdiOutput(it,:) = sFormation.spacecraftArray(1).gk;
                    detOutput(it,:) = sFormation.spacecraftArray(1).detect;
                    isoOutput(it,:) = sFormation.spacecraftArray(1).isolate;
                    it = it+1;
                    % Check if orbits explode
                    if any(euclidnorm(formationState(:,1:3)) - 1000*Re > 5E6 )
                        break
                    end

                end
                 %% Post Processing
                if faultType ~= 0 
                    faultTime = sFormation.spacecraftArray(satNo).faultTime;
                else
                    faultTime = 0.1;
                end
                faultVector = zeros(steps,1);
                faultVector(ceil(faultTime):end) = (satNo-1)*6+thrusterNo;
                detTime = sFormation.spacecraftArray(1).faultDetTime;
                parsave([fileName,num2str(ii),'_',num2str(faultType),'_',num2str(satNo),num2str(thrusterNo),'.csv'],[satNo,thrusterNo,faultType,faultParam,faultTime,detTime,faultVector',isoOutput']);
            end
        end
    end
end

