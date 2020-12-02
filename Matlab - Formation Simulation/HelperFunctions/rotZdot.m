function [rotMatrix] = rotZdot(angle)
%ROTZ Summary of this function goes here
%   Detailed explanation goes here
rotMatrix = [-sin(angle), cos(angle),0;
             -cos(angle), -sin(angle),0;
             0, 0, 0];
end
