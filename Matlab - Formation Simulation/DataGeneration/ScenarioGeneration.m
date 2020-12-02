clc;
close all;
clear all;
addpath('HelperFunctions')
addpath('DisturbanceForces')
addpath('Environment')
addpath('Environment/igrf')
addpath('Environment/igrf/datfiles')
addpath('FormationObjects')
%% Robustness Scenario Generation Script
highNoise = 1;
lowIntensity = 0;
lowThrust = 0;
multiFail = 0;
navChange = 0;
% setup
seed = 20;
rng(seed,'combRecursive')
nStart = 1;

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
noiseFactor = 1 ;

navArray = [];
if highNoise
    satMax = 6;
    thrustMax = 6;
    nEnd = 5;
    noiseFactor = 10;
    paramArray = (1/nEnd)*(1:nEnd);
    fileName = 'HighNoiseV3';
elseif lowIntensity
    satMax = 1;
    thrustMax = 1;
    nEnd = 100; 
    paramArray = logspace(-5,-1,nEnd);
    fileName = 'LowIntensityV2';
elseif lowThrust
    satMax = 1;
    thrustMax = 1;
    thrust = 0.04;
    nEnd = 100; 
    paramArray = (1/nEnd)*(1:nEnd);
    fileName = 'LowThrustV2';
elseif multiFail
    satMax = 6;
    thrustMax = 6;
    nEnd = 10;
    paramArray = 0.5+0.5/nEnd*(1:nEnd);
    fileName = 'MultiFail';
elseif navChange
    satMax = 6;
    thrustMax = 6;
    nEnd = 5;
    paramArray = (1/nEnd)*(1:nEnd);
    fileName = 'NavChangeV3';
    rangeNoiseSize = 1.1e-4;
    rangeBiasSize = 1.1e-4;
    angleNoiseSize = 9.8e-5;
    angleBiasSize = 9.8e-5;
    navArray = [rangeNoiseSize,rangeBiasSize,angleNoiseSize,angleBiasSize];
end
selfBiasSize = noiseFactor*1;
relBiasSize = noiseFactor*0.05;
velRelBiasSize = noiseFactor*0.0005;
selfNoise = noiseFactor*2;
relNoise = noiseFactor*0.02;
velRelNoise = noiseFactor*0.0005;

posErrWeight = 1e2; 
velErrWeight = 1e2;

load('covariance_Mean_Empirical.mat','Qr','resMean')

filePath = 'DataScenario/';
zeroed = true; 
parfor ii = nStart:nEnd
    stream = RandStream.getGlobalStream();
    stream.Substream = ii;
    for faultType = 1:2
        for satNo = 1:satMax
            for thrusterNo = 1:thrustMax
                if ~(multiFail && (satNo == 1 && thrusterNo == 1))
                    % Initial values
                    % Spacecraft Parameters 
                    spacecraftParameters = {mass, spacecraftSize, inertia, ...
                                            thrustInterval,burnTime,thrust,isp,mib...
                                            selfBiasSize, relBiasSize, velRelBiasSize...
                                            selfNoise, relNoise,velRelNoise,...
                                            posErrWeight,velErrWeight};


                    % Initialize formation
                    navType = navChange + 1 ;
                    sFormation = formation(n,type,center,velCenter,dist,formationOrientation,...
                                           controltype,disctype,spacecraftParameters,Qr,resMean,navType,navArray);
                    sFormation.disturbancesOn = 0;
                    faultTime = rand()*(0.5*T)+1000; % random time after settle down (300s) in the oscillation
                    faultParam = paramArray(ii);
                    if multiFail
                        sFormation.setFault(faultTime,1,1,faultType,faultParam)
                        sFormation.setFault(faultTime + 100,satNo,thrusterNo,faultType,faultParam)
                    else
                        sFormation.setFault(faultTime,satNo,thrusterNo,faultType,faultParam)
                    end

                    % Orbit time and time step
                    t0 = 0;
                    dt = Constants.dt;
                    nOrbit = 1;
                    tE = round(nOrbit*T/dt)*dt;

                    it= 1;
                    steps = round(tE/dt);
                    output = zeros((steps)*4*n+1,6);
                    offset = 1;

                    fdiOutput = zeros(steps,6*n);
                    detOutput = zeros(steps,1);
                    isoOutput = zeros(steps,1);
                    %% Integration
                    for t = t0:dt:tE-dt
                        % solve dynamics equation using rk4
                        sFormation.rk4Prop(t,dt)
                        formationState =  sFormation.getRelStates(zeroed);
                        output((it-1)*4*n+1 + offset:it*4*n + offset,1:6) = formationState;
                        fdiOutput(it,:) = sFormation.spacecraftArray(1).gk;
                        detOutput(it,:) = sFormation.spacecraftArray(1).detect;
                        isoOutput(it,:) = sFormation.spacecraftArray(1).isolate;
                        % Check if orbits explode
                        if any(euclidnorm(formationState(:,1:3)) - 1000*Re > 5E6 )
                            break
                        end
                        it = it+1;
                    end
                    if faultType == 1
                        faultTime = sFormation.spacecraftArray(satNo).faultTime;
                    end
                    output(1,1) = faultTime;
                    output(1,2) = satNo;
                    output(1,3) = thrusterNo;
                    output(1,4) = faultType;
                    output(1,5) = faultParam;

                    faultVector = zeros(steps,1);
                    faultVector(ceil(faultTime):end) = (satNo-1)*6+thrusterNo;
                    detTime = sFormation.spacecraftArray(1).faultDetTime;

                    fileCounter = ii;
                    %parsave([filePath,fileName,'_',num2str(faultType),'_',num2str(satNo),num2str(thrusterNo),'_',num2str(ii),'.csv'],output)
                    if multiFail
                        temp = fdiOutput';
                        kalmanDetailed = fdiOutput(:);
                        kalmanOutput = [satNo,thrusterNo,faultType,faultParam,faultTime,detTime,faultVector',isoOutput',kalmanDetailed'];
                    else
                        kalmanOutput = [satNo,thrusterNo,faultType,faultParam,faultTime,detTime,faultVector',isoOutput'];
                    end
                    parsave([filePath,'Kalman',fileName,'_',num2str(faultType),'_',num2str(satNo),num2str(thrusterNo),'_',num2str(ii),'.csv'],kalmanOutput)
                end
            end
        end
    end
end