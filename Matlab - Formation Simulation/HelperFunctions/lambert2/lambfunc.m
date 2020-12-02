function [statev, nsol] = lambfunc(ri, rf, tof, direct, revmax)

% solve Lambert's orbital two point boundary value problem

% Geza Gedeon's method

% input

%  ri     = initial position vector (kilometers)
%  rf     = final position vector (kilometers)
%  tof    = time of flight (seconds)
%  direct = transfer direction (1 = posigrade, -1 = retrograde)
%  revmax = maximum number of complete orbits

% output

%  nsol   = number of solutions
%  statev = matrix of state vector solutions of the
%           transfer trajectory after the initial delta-v

%  statev(1, sn) = position vector x component
%  statev(2, sn) = position vector y component
%  statev(3, sn) = position vector z component
%  statev(4, sn) = velocity vector x component
%  statev(5, sn) = velocity vector y component
%  statev(6, sn) = velocity vector z component
%  statev(7, sn) = semimajor axis
%  statev(8, sn) = orbital eccentricity
%  statev(9, sn) = orbital inclination
%  statev(10, sn) = argument of perigee
%  statev(11, sn) = right ascension of the ascending node
%  statev(12, sn) = true anomaly

%  where sn is the solution number

% Orbital Mechanics with MATLAB

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global mu ceps seps r1m r2m

pi2 = 2 * pi;

pidiv2 = 0.5 * pi;

xktab(1) = 1;

xktab(2) = -1;

sqrt2 = sqrt(2);

nsol = 0;

bkep1 = 0;

zerr = 0;

zsaved = 0;

% initialize state vector matrix

for j = 1:1:11

    for i = 1:1:12
        statev(i, j) = 0;
    end
end

% load "working" vectors

for i = 1:1:3
    r1(i) = ri(i);

    r2(i) = rf(i);

    chord(i) = r1(i) - r2(i);
end

r1sq = dot(r1, r1);

r1m = sqrt(r1sq);

r2sq = dot(r2, r2);

r2m = sqrt(r2sq);

r1dotr2 = dot(r1, r2);

ceps = r1dotr2 / r1m / r2m;

seps = sqrt(1 - ceps ^ 2);

r1xr2 = cross(r1, r2);

if ((r1xr2(3) * direct) < 0)
    seps = -seps;
end

eps = atan3(seps, ceps);

csq = dot(chord, chord);

c = sqrt(csq);

s = 0.5 * (r1m + r2m + c);

w2 = (r1m + r2m - c) / (2 * s);

w = sign(pi - eps) * sqrt(w2);

w3 = w2 * w;

ns = sqrt(mu / s ^ 3);

ms = ns * tof;

mpft = sqrt2 / 3 * (1 - w3);

del = sqrt2 / 100 * (1 - w3 * w2);

xk = 1;

lamda = 0;

if (ms - mpft <= 0)
    zmin = -1e20;

    zmax = 0;

    z0 = -1;

    if (abs(ms - mpft) < del)
        z0 = -0.0001;
    end

    nc = 2;
else
    zmin = 0;

    zmax = 1;

    z0 = 0.75;

    if (abs(ms - mpft) < del)
        z0 = 0.0001;
    end

    xp1 = (pidiv2 - asin(w) + w * sqrt(1 - w2)) / sqrt2;

    if (ms > xp1)
        xk = -1;
    end

    nc = 1;
end

z = z0;

[dndz, nz, z, zerr] = ziter(z, zmax, zmin, lamda, ms, nc, w, xk);

if (zerr == 0)
    [r1, verr] = vinitial(c, s, w, xk, z, r1, r2);

    if (verr == 1)
        return;
    end

    for ii = 1:1:3
        rr1(ii) = r1(ii);

        vr1(ii) = r1(ii + 3);
    end

    oev = eci2orb1(mu, rr1, vr1);

    if (abs(oev(2) - 1) <= 0.000001)
        oev(2) = 1;

        oev(1) = r1m * (1 + cos(oev(6)));

        nc = 3;
    end

    for ii = 1:1:6
        r1(6 + ii) = oev(ii);
    end

    for i = 1:1:12
        statev(i, 1) = r1(i);
    end

    nsol = 1;
end

if (revmax == 0 || nc ~= 1)
    return;
end

amin = s / 2;

pmin = pi2 * sqrt(amin ^ 3 / mu);

lammax = fix(tof / pmin);

if (lammax == 0)
    return;
end

tmin = xp1 / ns;

tmin1 = lammax * pmin + tmin;

if (tof < tmin1)
    niter = 1;

    z = 0.75;

    xk = 1;

    k = pi * (0.5 * (1 - xk) + lammax);

    while(1)
        az = abs(z);
        
        sqaz = sqrt(az);

        z1 = sqrt(1 - z);
        
        z2 = sqrt(1 - w2 * z);
        
        z3 = w * sqrt(az);

        f1 = asin(sqaz);
        
        f2 = asin(z3);

        nz = (k + xk * (f1 - z1 * az) - f2 + z3 * z2) / sqrt2 / z / sqaz;

        dndz = (xk / z1 - w3 / z2 - 3 / sqrt2 * nz) / z * (1 / sqrt2);

        if ((abs(dndz) <= 0.000001) || (abs(zsaved - z) < 0.00000001))
            break;
        end

        if (niter >= 50)
            lammax = lammax - 1;
            break;
        end

        dn2dz2 = (xk / z1 ^ 3 - w2 * w3 / z2 ^ 3 - 5 * sqrt2 * dndz) / 2 / sqrt2 / az;

        zsaved = z;

        z = z - dndz / dn2dz2;

        if (z > zmax)
            z = 0.5 * (zmax + zsaved);
        end

        if (z < zmin)
            z = 0.5 * (zmin + zsaved);
        end

        niter = niter + 1;

    end

    if (niter < 50)
        zl = z;
        
        tl = nz / ns;
        
        if ((tof - tl) <= 0)
            lammax = lammax - 1;
        else
            bkep1 = 1;
        end
    end
end

lammx = 5;

if (lammax < lammx)
    lammx = lammax;
end

if (revmax < lammx)
    lammx = revmax;
end

nsol = 2 * lammx + 1;

maxlam = lammx + 0.5;

if (maxlam == 0)
    return;
end

maxlm = fix(lammax + 0.5);

for lambda = 1:1:maxlam
    lamda = lambda;

    for k = 1:1:2
        xk = xktab(k);
        
        if ((lambda ~= maxlm) || (bkep1 ~= 1))
            % null
        else
            xk = 1;
            if (k == 1)
                zmax = zl;
                z0 = 0.75 * zl;
            else
                zmin = zl;
                zmax = 1;
                z0 = 0.5 * (zl + zmax);
            end
        end

        z = z0;

        [dndz, nz, z, zerr] = ziter(z, zmax, zmin, lamda, ms, nc, w, xk);

        if (zerr == 1)
            return;
        end

        [r1, verr] = vinitial(c, s, w, xk, z, r1, r2);

        if (verr == 1)
            return;
        end

        for ii = 1:1:3
            rr1(ii) = r1(ii);
            
            vr1(ii) = r1(ii + 3);
        end

        oev = eci2orb1(mu, rr1, vr1);

        for ii = 1:1:6
            r1(6 + ii) = oev(ii);
        end

        index = 2 * lambda + k - 1;

        for i = 1:1:12
            statev(i, index) = r1(i);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [r1, verr] = vinitial(c, s, w, xk, z, r1, r2)

global mu ceps seps r1m r2m

% initial velocity

verr = 0;

if (abs(seps) < 0.000001)
    verr = 1;
    
    return;
end

z1 = sqrt(1 - z);

z2 = sqrt(1 - z * w * w);

p = 2 / c * (s - r1m) * (s - r2m) * ((z2 + xk * w * z1) ^ 2) / c * s;

d = sqrt(mu) / sqrt(p) / seps;

f = d * (1 - ceps - p / r2m) / r1m;

g = d / r1m * p / r2m;

% compute initial velocity vector

for i = 1:1:3
    r1(i + 3) = f * r1(i) + g * r2(i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dndz, nz, z, zerr] = ziter (z, zmax, zmin, lamda, ms, nc, w, xk)

% z iteration

global mu ceps seps r1m r2m

niter = 0;

sqrt2 = sqrt(2);

oos2 = 1 / sqrt2;

zerr = 0;

zsaved = 2;

w2 = w * w;

w3 = w2 * w;

konst = pi * ((1 - xk) / 2 + lamda);

while(1)
    az = abs(z);
    
    sqaz = sqrt(az);

    z1 = sqrt(1 - z);
    
    z2 = sqrt(1 - w2 * z);
    
    z3 = w * sqaz;

    if (nc == 1)
        % elliptic orbit
        
        f1 = asin(sqaz);
        
        f2 = asin(z3);
    else
        % hyperbolic orbit
        
        f1 = log(sqaz + sqrt(sqaz ^ 2 + 1));
        
        f2 = log(z3 + sqrt(z3 * z3 + 1));
    end

    nz = (konst + xk * (f1 - z1 * sqaz) - f2 + z2 * z3) / sqrt2 / z / sqaz;

    dndz = (xk / z1 - w3 / z2 - 3 / sqrt2 * nz) / z * oos2;

    dn = ms - nz;

    if ((abs(z - zsaved) <= 0.00000001) || (abs(dn / ms) <= 0.00000001))
        return;
    end

    if (niter > 30)
        zerr = 1;
        break;
    end

    zsaved = z;

    z = z + dn / dndz;

    if (z > zmax)
        z = (zsaved + zmax) / 2;
    end

    if (z < zmin)
        z = (zsaved + zmin) / 2;
    end

    niter = niter + 1;

end


