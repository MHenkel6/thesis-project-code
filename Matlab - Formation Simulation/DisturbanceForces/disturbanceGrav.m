function distAcc = disturbanceGrav(time,posECI)
%DISTURBANCEGRAV Summary of this function goes here
%   Detailed explanation goes here

% Determine position of sun in ECI
% Find time and position just before current time
dT = Constants.Sun.deltaT;
tArr = Constants.Sun.tArray;

posSunArray = Constants.Sun.pos;
posMoonArray = Constants.Moon.pos;

t0 = floor(time/dT)*dT;
[~, ind0] = min(abs(tArr-t0));

posSun0= posSunArray(ind0,:);
posMoon0 = posMoonArray(ind0,:);
% Find time and position just after current time
t1 = ceil(time/dT)*dT;
[~, ind1] = min(abs(tArr-t1));
posSun1 = posSunArray(ind1,:);
posMoon1 = posMoonArray(ind1,:);
%Interpolate to get current position
posSun = posSun0 + (time-t0)/(dT)*(posSun1-posSun0);
posMoon = posMoon0 + (time-t0)/(dT)*(posMoon1-posMoon0);

dSun = posSun - posECI;
dMoon = posMoon - posECI;

accSun = Constants.muSun * (dSun/(norm(dSun)^3)- posSun/norm(posSun)^3);
accMoon = Constants.muMoon * (dMoon/(norm(dMoon)^3) - posMoon/norm(posMoon)^3);

distAcc = accSun+accMoon;
end

