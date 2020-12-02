function [r, v] = planet(ip, jdate)

% planetary ephemeris

% input

%  ip    = planet index (1 = mercury, 2 = venus, etc.)
%  jdate = julian day

% output

%  r = heliocentric position vector (kilometers)
%  v = heliocentric velocity vector (kilometers/second)

% Note: coordinates are with respect to the mean
%       ecliptic and equinox of date.

% Orbital Mechanics with Matlab

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global mu

% gravitational constant of the sun

mu = 132712441933;

% astronomical unit (kilometers)

aunit = 149597870;

% degrees to radians

dtr = pi / 180;

% time arguments

t = (jdate - 2451545) / 36525;

t2 = t * t;
t3 = t * t2;

% evaluate orbital elements for selected planet

switch ip
    case 1
        % mercury
        sma   =  0.387098310;
        ecc   =  0.20563175 + 0.000020406 * t - 0.0000000284 * t2 - 0.00000000017 * t3;
        inc   =  7.004986 + 0.0018215 * t - 0.00001809 * t2 + 0.000000053 * t3;
        mlong =  252.250906 + 149474.0722491 * t + 0.00030397 * t2 + 0.000000018 * t3;
        raan  =  48.330893 + 1.1861890 * t + 0.00017587 * t2 + 0.000000211 * t3;
        lper  =  77.456119 + 1.5564775 * t + 0.00029589 * t2 + 0.000000056 * t3;
    case 2
        % venus
        sma   =  0.723329820;
        ecc   =  0.00677188 - 0.000047766 * t + 0.0000000975 * t2 + 0.00000000044 * t3;
        inc   =  3.394622 + 0.0010037 * t - 0.00000088 * t2 - 0.000000007 * t3;
        mlong =  181.979801 + 58519.2130302 * t + 0.00031060 * t2 + 0.000000015 * t3;
        raan  =  76.679920 + 0.9011190 * t + 0.00040665 * t2 - 0.000000080 * t3;
        lper  =  131.563707 + 1.4022188 * t - 0.00107337 * t2 - 0.000005315 * t3;
    case 3
        % earth
        sma   =  1.000001018;
        ecc   =  0.01670862 - 0.000042037 * t - 0.0000001236 * t2 + 0.00000000004 * t3;
        inc   =  0;
        mlong =  100.466449 + 36000.7698231 * t + 0.00030368 * t2 + 0.000000021 * t3;
        raan  =  0;
        lper  =  102.937348 + 1.7195269 * t + 0.00045962 * t2 + 0.000000499 * t3;
    case 4
        % mars
        sma   =  1.523679342;
        ecc   =  0.09340062 + 0.000090483 * t - 0.0000000806 * t2 - 0.00000000035 * t3;
        inc   =  1.849726 - 0.0006010 * t + 0.00001276 * t2 - 0.000000006 * t3;
        mlong =  355.433275 + 19141.6964746 * t + 0.00031097 * t2 + 0.000000015 * t3;
        raan  =  49.558093 + 0.7720923 * t + 0.00001605 * t2 + 0.000002325 * t3;
        lper  =  336.060234 + 1.8410331 * t + 0.00013515 * t2 + 0.000000318 * t3;
    case 5
        % jupiter
        sma   =  5.202603191 + 0.0000001913 * t;
        ecc   =  0.04849485 + 0.000163244 * t - 0.0000004719 * t2 - 0.00000000197 * t3;
        inc   =  1.303270 - 0.0054966 * t + 0.00000465 * t2 - 0.000000004 * t3;
        mlong =  34.351484 + 3036.3027889 * t + 0.00022374 * t2 + 0.000000025 * t3;
        raan  =  100.464441 + 1.0209550 * t + 0.00040117 * t2 + 0.000000569 * t3;
        lper  =  14.331309 + 1.6126668 * t + 0.00103127 * t2 - 0.000004569 * t3;
    case 6
        % saturn
        sma   =  9.554909596 - 0.0000021389 * t;
        ecc   =  0.05550862 - 0.000346818 * t - 0.0000006456 * t2 + 0.00000000338 * t3;
        inc   =  2.488878 - 0.0037363 * t - 0.00001516 * t2 + 0.000000089 * t3;
        mlong =  50.077471 + 1223.5110141 * t + 0.00051952 * t2 - 0.000000003 * t3;
        raan  =  113.665524 + 0.8770979 * t - 0.00012067 * t2 - 0.000002380 * t3;
        lper  =  93.056787 + 1.9637694 * t2 + 0.00083757 * t2 + 0.000004899 * t3;
    case 7
        % uranus
        sma   =  19.218446062 - 0.0000000372 * t + 0.00000000098 * t2;
        ecc   =  0.04629590 - 0.000027337 * t + 0.0000000790 * t2 + 0.00000000025 * t3;
        inc   =  0.773196 + 0.0007744 * t + 0.00003749 * t2 - 0.000000092 * t3;
        mlong =  314.055005 + 429.8640561 * t + 0.00030434 * t2 + 0.000000026 * t3;
        raan  =  74.005947 + 0.5211258 * t + 0.00133982 * t2 + 0.000018516 * t3;
        lper  =  173.005159 + 1.4863784 * t + 0.00021450 * t2 + 0.000000433 * t3;
    case 8
        % neptune
        sma   =  30.110386869 - 0.0000001663 * t + 0.00000000069 * t2;
        ecc   =  0.00898809 + 0.000006408 * t - 0.0000000008 * t2 - 0.00000000005 * t3;
        inc   =  1.769952 - 0.0093082 * t - 0.00000708 * t2 + 0.000000028 * t3;
        mlong =  304.348665 + 219.8833092 * t + 0.00030926 * t2 + 0.000000018 * t3;
        raan  =  131.784057 + 1.1022057 * t + 0.00026006 * t2 - 0.000000636 * t3;
        lper  =  48.123691 + 1.4262677 * t + 0.00037918 * t2 - 0.000000003 * t3;
end

% argument of perihelion (radians)

argper = mod(dtr * (lper - raan), 2.0 * pi);

% mean anomaly (radians)

xma = mod(dtr * (mlong - lper), 2.0 * pi);

% solve Kepler's equation

[eanom, tanom] = kepler1(xma, ecc);

% load orbital elements array

oev(1) = aunit * sma;
oev(2) = ecc;
oev(3) = dtr * inc;
oev(4) = argper;
oev(5) = mod(dtr * raan, 2.0 * pi);
oev(6) = tanom;

% determine heliocentric state vector

[r, v] = orb2eci(mu, oev);


