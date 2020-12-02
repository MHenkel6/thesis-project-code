function [rotMatrix] = rotY(angle)
%ROTZ Summary of this function goes here
%   Detailed explanation goes here
rotMatrix = [cos(angle), 0,-sin(angle);
             0,1,0;
             sin(angle), 0,cos(angle)];
end

