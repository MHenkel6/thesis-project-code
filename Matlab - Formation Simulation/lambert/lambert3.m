% lambert3.m     December 16, 2012

% j2 perturbed solution of the Earth orbit Lambert problem

% shooting method with state transition matrix updates

% Orbital Mechanics with Matlab

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;

global mu j2 req rkcoef stmcoef

om_constants;

% initialize rkf78 integrator

rkcoef = 1;

% initialize state transition matrix algorithm

stmcoef = 1;

clc; home;

fprintf('\n              program lambert3\n');

fprintf('\n< j2 perturbed Earth orbit lambert problem >\n\n');

% begin simulation

fprintf('\ninitial orbit\n');

oev1 = getoe([1;1;1;1;1;1]);

fprintf('\n\nfinal orbit \n');

oev2 = getoe([1;1;1;1;1;1]);

% transfer time

while(1)
   fprintf('\n\nplease input the transfer time in minutes\n');
   
   ttmins = input('? ');
   
   if (ttmins > 0.0)
      break;
   end
end   

% time of flight (seconds)

tof = 60.0 * ttmins;

direct = 1;

revmax = 0;

% compute state vectors of initial and final orbits

[ri, vi] = orb2eci(mu, oev1);
      
[rf, vf] = orb2eci(mu, oev2);

% save the final position vector

rfsaved = rf;

% compute initial guess for delta-v vector

for i = 1:1:3
    sv1(i) = ri(i);
    sv1(i + 3) = vi(i);
    
    sv2(i) = rf(i);
    sv2(i + 3) = vf(i);
end

[vito, vfto] = glambert(mu, sv1, sv2, tof, revmax);

v1 = vito';

oev3 = eci2orb1(mu, ri, v1);

% initial guess for delta-v (kilometers/second)

dv(1) = vito(1) - vi(1);
dv(2) = vito(2) - vi(2);
dv(3) = vito(3) - vi(3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% determine j2-perturbed solution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tetol = 1.0e-8;

neq = 6;

niter = 0;

while(1)
    
   niter = niter + 1;
     
   % compute state transition matrix
   
   [rf, vf, stm] = stm2 (mu, tof, ri, v1);
   
   % extract velocity sub-matrix
   
   stm12(1:3, 1:3) = stm(1:3, 4:6);
      
   % load current state vector
   
   x(1) = ri(1);
   x(2) = ri(2);
   x(3) = ri(3);
   
   x(4) = v1(1);
   x(5) = v1(2);
   x(6) = v1(3);
   
   h = 10.0;
   
   ti = 0.0;
   
   tf = tof;
   
   % integrate equations of motion
   
   xout = rkf78 ('j2eqm', neq, ti, tf, h, tetol, x);
   
   % calculate position vector errors
   
   drf(1) = rfsaved(1) - xout(1);
   drf(2) = rfsaved(2) - xout(2);
   drf(3) = rfsaved(3) - xout(3);
   
   % check for convergence
   
   drss = norm(drf);
   
   if (drss < 0.0000001 || niter > 10)
      break;
   end
   
   % compute delta-v correction
   
   dv = inv(stm12) * drf';
   
   % update initial velocity vector
   
   v1 = v1 + dv;

end

% compute delta-v components and magnitude
   
dv(1) = v1(1) - vi(1);
dv(2) = v1(2) - vi(2);
dv(3) = v1(3) - vi(3);

dvm = norm(dv);

fprintf('\n                    program lambert3\n');

fprintf('\n        j2 perturbed Earth orbit lambert problem \n');

fprintf('\n  shooting method with state transition matrix updates \n\n');

fprintf('\norbital elements of the initial orbit\n');

oeprint1(mu, oev1);

fprintf('\n\norbital elements of the final orbit \n');

oeprint1(mu, oev2);

fprintf('\n\nkeplerian transfer orbit\n');

oeprint1(mu, oev3);

fprintf('\n\nj2 perturbed transfer orbit\n');

oev4 = eci2orb1 (mu, ri', v1');

oeprint1(mu, oev4);

fprintf('\n\ndelta-v vector and magnitude\n');
   
fprintf('\nx-component of delta-v      %10.4f  meters/second', 1000.0 * dv(1));

fprintf('\ny-component of delta-v      %10.4f  meters/second', 1000.0 * dv(2));

fprintf('\nz-component of delta-v      %10.4f  meters/second', 1000.0 * dv(3));

fprintf('\n\ntotal delta-v               %10.4f  meters/second', 1000.0 * dvm);
   
fprintf('\n\ntransfer time               %10.4f  minutes \n', ttmins);

fprintf('\n\nfinal position vector error components and magnitude\n');

fprintf('\nx-component of delta-r      %12.8f  meters', 1000.0 * drf(1));

fprintf('\ny-component of delta-r      %12.8f  meters', 1000.0 * drf(2));

fprintf('\nz-component of delta-r      %12.8f  meters', 1000.0 * drf(3));

fprintf('\n\ndelta-r magnitude           %12.8f  meters\n\n', 1000.0 * norm(drf));


