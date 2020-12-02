function [dev] = nonLinRel(state,formCenter,velCenter,a,e,argLat)

x =  state(1);
y =  state(2);
z =  state(3);
xdot =  state(4);
ydot =  state(5);
zdot =  state(6);
r = norm(formCenter);
mu = Constants.muEarth;
n = sqrt(Constants.muEarth/(a^3*(1-e^2)^3))*(1+e*cos(argLat))^2;
rdot = norm(dot(formCenter,velCenter)/norm(formCenter));
ndot = -2*rdot*n/r;
% Set velocities
dev(1:3) = [xdot,ydot,zdot];
% Set accelerations
denom = ((r+x)^2 + y^2 + z^2)^(3/2);
% Radial
dev(4) = mu/r^2 - mu*(r+x)/denom + 2*n*ydot + ndot*y + n^2*x;
% Along Track
dev(5) = -mu*y/denom - 2*n*xdot - ndot*x + n^2*y ; 
% Cross Track
dev(6) = -mu*z/denom;

end

