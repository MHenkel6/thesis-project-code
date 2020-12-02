function [ stateChange ] = dynamics(time,posECI, velECI,mass,cF,disturbancesOn)%
%% Compute derivatives of state
% Inputs:
% time
% posECI
% velECI
% spacecraft
% cF
% disturbancesOn
% Outputs:
% stateChange       = Derivative of states
%
% Author: Martin Henkel
%%
stateChange = zeros(1,6);

%Determine guidance Delta V
%deltaV = spacecraft.guidance(time,posECI, velECI);
stateChange(1:3) = velECI;

%Determine forces acting on body
% Transform to ECEF
rotECEF = rotZ(Constants.planetRot*time);
posECEF = rotECEF*posECI';
[gx,gy,gz] = gravityCustom(posECEF');
g = rotECEF\[gx,gy,gz]';
Fg = g'*mass;
% Disturbances
if disturbancesOn
    Frad = disturbanceRadiation(time, posECI);
    Fdrag = disturbanceDrag(posECI,velECI);
    Fmag = disturbanceMag(time,posECEF,velECI, Constants.spacecraftCharge );
    Fgrav = disturbanceGrav(time,posECI)*mass;
    % Control Forces
    %sum of forces
%     if Constants.highDisturbances
%         distFactor = 10000;
%     else
%         distFactor = 1;
%     end
    Fdist = (Frad + Fdrag + Fgrav + Fmag);
    F = Fg + Fdist;
else
    F = Fg;
end

stateChange(4:6) = (F+cF)/mass;
end
