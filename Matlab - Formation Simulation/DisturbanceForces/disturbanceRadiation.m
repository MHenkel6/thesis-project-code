function distForce = disturbanceRadiation(time,posECI)
%DISTURBANCESOLAR Determine disturbance force from solar wind and radiation
% Three components: solar radiation, earth reflection and earth infrared


% Determine position of sun in ECI
% Find time and position just before current time
dT = Constants.Sun.deltaT;
tArr = Constants.Sun.tArray;
t0 = floor(time/dT)*dT;
posSunArray = Constants.Sun.pos;

[~, ind0] = min(abs(tArr-t0));
posSun0= posSunArray(ind0,:);
% Find time and position just after current time
t1 = ceil(time/dT)*dT;
[~, ind1] = min(abs(tArr-t1));
posSun1 = posSunArray(ind1,:);
%Interpolate to get current position
posSun = posSun0 + (time-t0)/(dT)*(posSun1-posSun0);
% Direction vectors
dirSun = (posECI-posSun)/norm(posECI-posSun); % Vector from Sun to Satellite
dirEarth = posECI/norm(posECI); % Vector from Earth to Satellite
dirSunEarth = posSun/norm(posSun); % Vector from Earth to Sun 

Cr = 0.9; % reflectiviy of aluminum in visible sprectum
% taken frmo https://laserbeamproducts.wordpress.com/2014/06/19/reflectivity-of-aluminium-uv-visible-and-infrared/


% Check if in shadow (assume parallel sun rays)
projDist = norm(posECI -dirSunEarth*dot(dirSunEarth,posECI));%projected distance on earth circle
if (projDist<6371000) && (angleVec(dirSunEarth,dirEarth)>pi/2)% If distance normal to Earth-Sun vector is less than Earth radius -> shadow
    Wsun = 0;
    Wreflect = 0;
else
    Wsun = 1361/(1+0.0334*cos(2*pi*75/365)); % Solar intensity in LEO
    % Reflected Sunlight
    Wreflect = 0.32 * Wsun*cos(angleVec(dirSunEarth,dirEarth));
end

Wearth = 240;
A = 3*sqrt(3)/2; % Effective crossectional area, update later
c = 299792458; % speed of light in m/s
% f = Cr W A /c from Wakker p. 542
distForce = Cr*A/c*(Wsun*dirSun+(Wearth+Wreflect)*dirEarth) ;
end

