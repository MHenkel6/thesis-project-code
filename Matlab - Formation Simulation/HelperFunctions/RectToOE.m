function OEVec = RectToOE(rectState)
%RectToOE Transformation from rectangular coordinates to orbital Elements
%   
r = sqrt(sum(rectState(1:3).^2);
v = sqrt(sum(rectState(4:6).^2);

a = r/(2-r*v^2/Constants.muEarth);
e;
inc;
argPeri;
RAAN;
theta;
OEVec = [a;
         e;
         inc;
         RAAN;
         argPeri;
         theta];
end

