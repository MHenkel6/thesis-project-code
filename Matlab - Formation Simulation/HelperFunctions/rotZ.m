function [rotMatrix] = rotZ(angle)
%ROTZ Summary of this function goes here
%   Detailed explanation goes here
rotMatrix = [cos(angle), sin(angle),0;
             -sin(angle), cos(angle),0;
             0, 0, 1];
end

