function Fmag = disturbanceMag(time,posECEF,velECI,q)
%DISTURBANCEMAG Summary of this function goes here
%   Detailed explanation goes here
[lat,lon,alt] = ecef2geod(posECEF(1),posECEF(2),posECEF(3));
tmag = datenum(2008,9,17,0,0,time);
[Bx,By,Bz] = igrf(tmag, lat,lon,alt/1000, 'geodetic');
B = rotY(pi/2)*rotX(lat)*rotY(lat)*[Bx,By,Bz]'/10^9;

Fmag = cross(velECI*q,B);
end

