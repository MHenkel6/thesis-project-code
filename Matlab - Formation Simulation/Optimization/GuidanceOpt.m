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
addpath('../snopt-matlab-3.0.0/matlab')
addpath('../snopt-matlab-3.0.0/matlab/util')
addpath('../snopt-matlab-3.0.0/mex')
%% Setup
% Initial values
global printDV;
printDV = 0;
GM = Constants.muEarth;
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
x0 = [1,0,0,0,0.01,0.5,1];
A = [];
b = [];
Aeq = [];
beq = [];
lb= [-Inf,-Inf,-Inf,-Inf,0,0,0];
ub = [Inf,Inf,Inf,Inf,1,1,1];
nonlcon = @quatCon;
x = fmincon(@cost,x0,A,b,Aeq,beq,lb,ub,nonlcon,options)
%%
printDV = 1;
minCost= cost(x)
%}
distMin = 100;
distMax = 1000;

hMin = 350;
hMax = 1000;

incMin = 0;
incMax = pi/2;
dist = distMin + (distMax-distMin)*x(5)
h = hMin + (hMax-hMin)*x(6)
inc = 180/pi*(incMin + (incMax-incMin)*x(7))
StateBeforeMin = [0.9328,-0.0854,0.0314,0.3488,0.0471,0.5192, 0.6979];
cBefore = cost(StateBeforeMin)
% 'State' for cost function and optimization should be the orbital
% parameters of the reference orbit and the orientation and offset of the
% formation orientation. 
%% Define cost function
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
    
    qOrientation = state(1:4);
    dist = distMin + (distMax-distMin)*state(5);
    h = hMin + (hMax-hMin)*state(6);
    inc = incMin + (incMax-incMin)*state(7);
    
    w1 = 0.5;
    w2 = 0.5;
    
  
    distCenter = dist*sqrt(2)/2; %distance from center
    distArray = [0,0,distCenter;
                 distCenter,0,0;
                 0,distCenter,0;
                 -distCenter,0,0;
                 0 -distCenter,0;
                 0,0,-distCenter];
             
   
    GM = Constants.muEarth;
    Re = 6371;
    
    
    center = [Re+h,0,0]'*1000;
    muEarth = 3.986E14;
    T = 2*pi*sqrt(((Re+h)*1000)^3/muEarth);
    nImpulse = round(T/60);
    velCenter =[0,sqrt(GM/(norm(center)))*cos(inc),sqrt(GM/(norm(center)))*sin(inc)]'; % Reference velocity of center point
    % Determine Center Reference Positions
    
    dtArr = 0:T/nImpulse:T;
    [a,eMag,inc,O,o,nu,truLon,argLat,lonPer,p] = rv2orb(center,velCenter);
    [pos,vel] = keplerEQsolve(a,eMag,inc,O,o,nu,truLon,argLat,lonPer,p,dtArr);
    posArray = zeros(6*nImpulse,3);
    % Determine reference positions of each satellite
    for ii = 1:size(pos,1)
        velDir = vel(ii,:)/norm(vel(ii,:));
        refPos = pos(ii,:);
        posArray(6*(ii-1)+1:6*ii,:) = refPos+quatrotate(qOrientation,distArray);
    end    
    % Determine delta V for each satellite and positions
    dVArray = zeros(6,1);
    dVvecArray = zeros(6*nImpulse,3);
    for jj = 1:6
        for ii = 1:nImpulse
            posA = posArray(6*(ii-1)+jj,:);
            posB = posArray(6*(ii)+jj,:);
            velA = vel(ii,:);
            velB = vel(ii+1,:);
            [v1B,v2B] = lambertBook(posA,posB,T/(nImpulse),'pro');
           
            %[v1,v2] = lambert(posA/1000,posB/1000,T/(nImpulse),0,Constants.muEarth);
            if ii == 1
                velNext = vel(ii,:);
            end
            dVvecArray(ii+nImpulse*(jj-1),:) = v1B-velNext;
            deltaV = norm((v1B-velNext));%+norm((v2B*1000-vel(ii+1,:)));
            dVArray(jj) = dVArray(jj)+deltaV;
            velNext = v2B;
        end
    end
    
    % Calculate cost from DeltaV and positions
    Q= 0;
    C = w1*mean(dVArray)/0.9*(3600/T)+w2*std(dVArray)/mean(dVArray);
    if printDV == 1
        dVArray
        save('deltaV','dVvecArray');
    end
end

function [c,ceq] = quatCon(state)
% Function calculating quaternion constraint
quat = state(1:4);
c = [];
ceq= sqrt(sum(quat.^2))-1;
end