function [ vecOutput ] = vecInterpolate( x1,vecInput,x2)
%% Linearly interpolates nx3 vector array
% Inputs:
% x1        = nx1 spacing of inputarray vecInput
% vecInput  = nx3 array spaced by x1
% x2        = mx1 spacing of outputarray 
% Outputs:
% vecOutput = mx3 linearly interpolated vector
% Author : Martin
%%
% Interpolate each variabe independently
v1 = interp1(x1,vecInput(:,1),x2); 
v2 = interp1(x1,vecInput(:,2),x2);
v3 = interp1(x1,vecInput(:,3),x2);
vecOutput = [v1;v2;v3]';
end

