function [ stateChange ] = rotDynamics(state,inertia)%
%% Compute derivatives of state
% Inputs:
% state             = 1x7 state vector, 1:4 attitude quaternions - [-], 5:7
%                     angular rates - [rad/s]
% Bdot              = Time derivative of Magnetic field vector in Body
%                     frame - [nT/s]
% positionSun       = position of the Sun in ECI - [m] 
% positionDelfi     = position of Delfi in ECI - [m] 
% velocityDelfi     = velocity of Delfi in ECI - [m/s]
% airDensity        = Density of Air - [kg/m^3]
% resMomentDelfi    = residual Magnetic Dipole Moment - [A m^2]
% BInertial         = Mangetic field vector in inertial Frame - [nT]
% Outputs
% stateChange       = Derivative of states
%
% Author: Martin Henkel
%%
stateChange = zeros(7,1);
torques = [0;
           0;
           0]; % 
%For clarity, get angular rate vector omega
omega  = state(5:7); 
% inertia= Constants.inertiaTensor; % - input
%Skew Symmetric omega matrix
S = [0       ,-omega(3),omega(2);
     omega(3),0        ,-omega(1);
     -omega(2),omega(1) ,0];
%Quaternion matrix representation  of angular rate 
% NOTE: quaternion scalar in Matlab is in index 0
OmegaQuat = [0         ,-omega(1),-omega(2) ,-omega(3);
            omega(1)   ,0        , omega(3) ,-omega(2);
            omega(2)   ,-omega(3),0         , omega(1);
            omega(3)   , omega(2),-omega(1) ,0   ];
%Kinemtatics, propagating quaternion states    
stateChange(1:4) =0.5*OmegaQuat*state(1:4);%qdot = 1/2*OMEGA*q
%Dynamics (Euler Equation), propagating angular rates
stateChange(5:7) = -inertia\(S*inertia*omega-torques);
end
