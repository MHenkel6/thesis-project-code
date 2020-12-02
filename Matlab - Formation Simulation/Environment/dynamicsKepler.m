function [ stateChange ] = dynamicsKepler(posECI, velECI)%
%% Compute derivatives of state
% Inputs:
% state             = 1x7 state vector, 1:4 attitude quaternions - [-], 5:7
%                     angular rates - [rad/s]
% Outputs:
% stateChange       = Derivative of states
%
% Author: Martin Henkel
%%

stateChange = zeros(1,6);
stateChange(1:3) = velECI;
%Determine forces acting on body
accG = -Constants.muEarth*posECI/norm(posECI)^3;
stateChange(4:6) = accG;
end
