function [pos,vel,theta] = keplerEQsolve(a,e,i,O,o,nu,truLon,argLat,lonPer,p,dT)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
pos = zeros(size(dT,2),3);
vel = zeros(size(dT,2),3);
% Calculate mean anomaly;
it = 1;

for t = dT
    M = sqrt(Constants.muEarth/a^3)*t;
    % Solve Kepler Equation 
    kepEq = @(E)E-e*sin(E)-M;
    options = optimset('Display','off');
    Esol = fsolve(kepEq,M,options);
    % Detemine true anomaly
    theta = 2*atan(sqrt((1 + e)/(1 - e))*tan(Esol/2));
    
    % Determine positions in ECI
    [pos(it,:),vel(it,:)] = orb2rv(p,e,i,O,o,nu+theta,truLon+theta,argLat+theta,lonPer);
    it = it+1;
end
pos = real(pos);
vel = real(vel);
end

