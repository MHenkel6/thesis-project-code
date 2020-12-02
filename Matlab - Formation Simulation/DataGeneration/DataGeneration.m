clc;
close all;
clear all;
addpath('HelperFunctions')
addpath('DisturbanceForces')
addpath('Environment')
addpath('Environment/igrf')
addpath('Environment/igrf/datfiles')
addpath('FormationObjects')
%% Data Generation Script

% setup
seed = 19;
rng(seed,'combRecursive')
nStart = 900;
nEnd = 900; 
% Loop
Re = Constants.graviPar.Re/1000;
h = 687.4800;
inc =  62.8110/360*2*pi;

n = 6; 
type = 'octahedron';
center = [Re+h,0,0]*1000;
GM = Constants.muEarth;
velCenter = [0,sqrt(GM/(norm(center)))*cos(inc),sqrt(GM/(norm(center)))*sin(inc)]; % Reference velocity of center point
muEarth = 3.986E14;
T = 2*pi*sqrt(((Re+h)*1000)^3/muEarth);
dist = 142.3900;
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
velRelBiasSize = 0.0005;
selfNoise = 2;
relNoise = 0.02;
velRelNoise = 0.0005;
posErrWeight = 1e5; 
velErrWeight = 1e5;

fileName = 'Data900/DataOpenFault4N_';%'DataII/DataNoFault4N_';%'Data/DataNoFault_';
zeroed = true; 
parfor ii = nStart:nEnd
    stream = RandStream.getGlobalStream();
    stream.Substream = ii;
    for satNo = 1:6
        for thrusterNo = 1:6
            % Initial values
            % Spacecraft Parameters 
            spacecraftParameters = {mass, spacecraftSize, inertia, ...
                                    thrustInterval,burnTime,thrust,isp,mib...
                                    selfBiasSize, relBiasSize, velRelBiasSize...
                                    selfNoise, relNoise,velRelNoise,...
                                    posErrWeight,velErrWeight};


            % Initialize formation
            sFormation = formation(n,type,center,velCenter,dist,formationOrientation,...
                                   controltype,disctype,spacecraftParameters);
            sFormation.disturbancesOn = 0;
            faultType = 2; % faulttype, 1 is closed, 2 is open
            faultTime = rand()*(0.5*T)+300; % random time after settle down (300s) in the oscillation
            faultParam = rand()*0.9 + 0.1;
            sFormation.setFault(faultTime,satNo,thrusterNo,faultType,faultParam)
            % Orbit time and time step
            t0 = 0;
            dt = Constants.dt;
            nOrbit = 1;
            tE = round(nOrbit*T/dt)*dt;

            it= 1;
            steps = round(tE/dt);
            output = zeros((steps)*4*n+1,6);
            offset = 1;
            
            %% Integration
            for t = t0:dt:tE-dt
                % solve dynamics equation using rk4
                sFormation.rk4Prop(t,dt)
                formationState =  sFormation.getRelStates(zeroed);
                output((it-1)*4*n+1 + offset:it*4*n + offset,1:6) = formationState;

                % Check if orbits explode
                if any(euclidnorm(formationState(:,1:3)) - 1000*Re > 5E6 )
                    break
                end
                it = it+1;
            end
            
            output(1,1) = sFormation.spacecraftArray(satNo).faultTime;
            output(1,2) = satNo;
            output(1,3) = thrusterNo;
            output(1,4) = faultType;
            output(1,5) = faultParam;
            
            fileCounter = nStart + 36*(ii-nStart)+6*(satNo-1)+thrusterNo;
            parsave([fileName,num2str(satNo),num2str(thrusterNo),'_',num2str(ii),'.csv'],output)
        end
    end
end