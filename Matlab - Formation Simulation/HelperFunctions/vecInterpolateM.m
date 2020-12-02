function [ vecOutput ] = vecInterpolate( x1,vecInput,x2)
%% Linearly interpolates nx3 vector array
% Inputs:
% x1        = nx1 spacing of inputarray vecInput
% vecInput  = nxm array spaced by x1
% x2        = kx1 spacing of outputarray 
% Outputs:
% vecOutput = kx3 linearly interpolated vector
% Author : Martin
%%
% Interpolate each variabe independently
vecOutput = [];
for ii = 1:size(vecInput,2)
    vii =  interp1(x1,vecInput(:,ii),x2); 
    vecOutput = [vecOutput;vii];
end
vecOutput = vecOutput';
end

