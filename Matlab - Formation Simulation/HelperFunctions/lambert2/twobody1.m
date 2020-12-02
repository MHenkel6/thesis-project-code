function [rf, vf] = twobody1 (mu, tau, ri, vi)

% solve the two body initial value problem

% Goodyear's method
   
% input

%  mu  = gravitational constant (km**3/sec**2)
%  tau = propagation time interval (seconds)
%  ri  = initial eci position vector (kilometers)
%  vi  = initial eci velocity vector (km/sec)

% output

%  rf = final eci position vector (kilometers)
%  vf = final eci velocity vector (km/sec)

% Orbital Mechanics with Matlab

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global tbcoef

global a0 b0 c0 d0 e0 f0 g0 h0 i0 j0 k0

global l0 m0 n0 o0 p0

if (tbcoef == 1)

   % define coefficients
      
   a0 = 0.025;
   b0 = a0 / 42;
   c0 = b0 / 72;
   d0 = c0 / 110;
   e0 = d0 / 156;
   f0 = e0 / 210;
   g0 = f0 / 272;
   h0 = g0 / 342;
   i0 = 1 / 24;
   j0 = i0 / 30;
   k0 = j0 / 56;
   l0 = k0 / 90;
   m0 = l0 / 132;
   n0 = m0 / 182;
   o0 = n0 / 240;
   p0 = o0 / 306;

   tbcoef = 0;
end

% convergence criterion

tol = 1.0e-8;

rsdvs = dot(ri, vi);

rsm = norm(ri);

vsm2 = dot(vi, vi);

zsma = 2.0 / rsm - vsm2 / mu;

if (zsma > 0.0)
   psi = tau * zsma;
else
   psi = 0.0;
end

alp = vsm2 - 2.0 * mu / rsm;

for z = 1:1:20
    m = 0;

    psi2 = psi * psi;

    psi3 = psi * psi2;

    aas = alp * psi2;

    if (aas ~= 0) 
       zas = 1 / aas;
    else
       zas = 0;
    end

    while (abs(aas) > 1)
       m = m + 1;
       aas = 0.25 * aas;
    end

    pc5 = a0 + (b0 + (c0 + (d0 + (e0 + (f0 + (g0 + h0 * aas) * aas) ...
          * aas) * aas) * aas) * aas) * aas;
    pc4 = i0 + (j0 + (k0 + (l0 + (m0 + (n0 + (o0 + p0 * aas) * aas) ...
          * aas) * aas) * aas) * aas) * aas;
    pc3 = (0.5 + aas * pc5) / 3;
    pc2 = 0.5 + aas * pc4;
    pc1 = 1 + aas * pc3;
    pc0 = 1 + aas * pc2;

    if (m > 0)
       while (m > 0)
          m = m - 1;
          pc1 = pc0 * pc1;
          pc0 = 2 * pc0 * pc0 - 1;
       end
       
       pc2 = (pc0 - 1) * zas;
       pc3 = (pc1 - 1) * zas;
    end

    s1 = pc1 * psi;
    s2 = pc2 * psi2;
    s3 = pc3 * psi3;

    gg = rsm * s1 + rsdvs * s2;

    dtau = gg + mu * s3 - tau;

    rfm = abs(rsdvs * s1 + mu * s2 + rsm * pc0);
       
    if (abs(dtau) < abs(tau) * tol)
       break;
    else
       psi = psi - dtau / rfm;
    end
end

rsc = 1 / rsm;

r2 = 1 / rfm;

r12 = rsc * r2;

fm1 = -mu * s2 * rsc;

ff = fm1 + 1;

fd = -mu * s1 * r12;

gdm1 = -mu * s2 * r2;

gd = gdm1 + 1;

% compute final state vector

for i = 1:1:3
    rf(i) = ff * ri(i) + gg * vi(i);
    
    vf(i) = fd * ri(i) + gd * vi(i);
end


