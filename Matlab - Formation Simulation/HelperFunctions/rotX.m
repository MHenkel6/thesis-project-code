function [rotMatrix] = rotX(angle)
%ROTZ Summary of this function goes here
%   Detailed explanation goes here
rotMatrix = [1,0,0;
             0,cos(angle) ,sin(angle);
             0,-sin(angle),cos(angle)];
end

