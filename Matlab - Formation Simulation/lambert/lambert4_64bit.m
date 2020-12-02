% lambert4_64bit.m     July 8, 2014

% NLP solution of the perturbed Earth orbit Lambert problem

% 64 bit SNOPT algorithm (Mar 17, 2014 version)

% flyby and rendezvous trajectory types

% Orbital Mechanics with MATLAB

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;

global mu j2 req neq tetol rkcoef

global otype ri vi tof rtarget vtarget drf dvf

om_constants;

% initialize rkf78 algorithm

rkcoef = 1;

% rkf78 convergence tolerance

tetol = 1.0e-8;

% number of differential equations

neq = 6;

clc; home;

fprintf('\n             program lambert4\n');

fprintf('\n< perturbed Earth orbit Lambert problem >\n\n');

while(1)
    
    fprintf('\ntrajectory type (1 = flyby, 2 = rendezvous)\n');
    
    otype = input('? ');
    
    if (otype == 1 || otype == 2)
        break;
    end
    
end

fprintf('\n\nclassical orbital elements of the initial orbit\n');

oev1 = getoe([1;1;1;1;1;1]);

fprintf('\n\nclassical orbital elements of the final orbit \n');

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

revmax = 0;

% compute state vectors of initial and final orbits

[ri, vi] = orb2eci(mu, oev1);

[rf, vf] = orb2eci(mu, oev2);

% save the final position and velocity vectors

rtarget = rf;

vtarget = vf;

% compute initial guess for delta-v vector
% using Gooding Lambert algorithm

for i = 1:1:3
    
    sv1(i) = ri(i);
    
    sv1(i + 3) = vi(i);
    
    sv2(i) = rf(i);
    
    sv2(i + 3) = vf(i);
    
end

[vito, vfto] = glambert(mu, sv1, sv2, tof, revmax);

oev3 = eci2orb1(mu, ri, vito');

% initial guess for delta-v vectors (kilometers/second)

xg(1) = vito(1) - vi(1);
xg(2) = vito(2) - vi(2);
xg(3) = vito(3) - vi(3);

if (otype == 2)
    
    % rendezvous
    
    xg(4) = vf(1) - vfto(1);
    xg(5) = vf(2) - vfto(2);
    xg(6) = vf(3) - vfto(3);
    
end

xg = xg';

fprintf('\n\ntwo-body guess for initial delta-v vector and magnitude\n');

fprintf('\nx-component of delta-v      %12.6f  meters/second', 1000.0 * xg(1));

fprintf('\ny-component of delta-v      %12.6f  meters/second', 1000.0 * xg(2));

fprintf('\nz-component of delta-v      %12.6f  meters/second', 1000.0 * xg(3));

fprintf('\n\ndelta-v magnitude           %12.6f  meters/second\n', 1000.0 * norm(xg));

if (otype == 2)
    
    fprintf('\n\ntwo-body guess for final delta-v vector and magnitude\n');
    
    fprintf('\nx-component of delta-v      %12.6f  meters/second', 1000.0 * xg(4));
    
    fprintf('\ny-component of delta-v      %12.6f  meters/second', 1000.0 * xg(5));
    
    fprintf('\nz-component of delta-v      %12.6f  meters/second', 1000.0 * xg(6));
    
    fprintf('\n\ndelta-v magnitude           %12.6f  meters/second', 1000.0 * norm(xg(4:6)));
    
    fprintf('\n\ntotal delta-v               %12.6f  meters/second\n', ...
        1000.0 * (norm(xg(1:3)) + norm(xg(4:6))));
    
end

% define lower and upper bounds for components of delta-v vectors (kilometers/second)

for i = 1:1:3
    
    xlwr(i) = min(-1.1 * norm(xg(1:3)), -75.0);
    
    xupr(i) = max(+1.1 * norm(xg(1:3)), +75.0);
    
end

if (otype == 2)
    
    % rendezvous
    
    for i = 4:1:6
        
        xlwr(i) = min(-1.1 * norm(xg(4:6)), -75.0);
        
        xupr(i) = max(+1.1 * norm(xg(4:6)), +75.0);
        
    end
    
end

xlwr = xlwr';

xupr = xupr';

% bounds on objective function

flow(1) = 0.0d0;

fupp(1) = +Inf;

% enforce final position vector equality constraints

flow(2) = 0.0d0;
fupp(2) = 0.0d0;

flow(3) = 0.0d0;
fupp(3) = 0.0d0;

flow(4) = 0.0d0;
fupp(4) = 0.0d0;

if (otype == 2)
    
    % rendezvous - enforce final velocity vector equality constraints
    
    flow(5) = 0.0d0;
    fupp(5) = 0.0d0;
    
    flow(6) = 0.0d0;
    fupp(6) = 0.0d0;
    
    flow(7) = 0.0d0;
    fupp(7) = 0.0d0;
    
end

flow = flow';

fupp = fupp';

if (otype == 1)
    
    xmul = zeros(3, 1);
    
    xstate = zeros(3, 1);
    
    fmul = zeros(4, 1);
    
    fstate = zeros(4, 1);
    
else
    
    xmul = zeros(6, 1);
    
    xstate = zeros(6, 1);
    
    fmul = zeros(7, 1);
    
    fstate = zeros(7, 1);
    
end

% solve the orbital TPBVP using SNOPT

snscreen on;

[x, f, inform, xmul, fmul] = snopt(xg, xlwr, xupr, xmul, xstate, ...
    flow, fupp, fmul, fstate, 'tpbvp');

snprint('off');

snsummary('off');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% predict the final mission orbit
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xi(1) = ri(1);
xi(2) = ri(2);
xi(3) = ri(3);

xi(4) = vi(1) + x(1);
xi(5) = vi(2) + x(2);
xi(6) = vi(3) + x(3);

% initial guess for step size (seconds)

h = 10.0;

% initial time (seconds)

ti = 0.0;

% final time (seconds)

tf = tof;

% integrate equations of motion

xf = rkf78 ('j2eqm', neq, ti, tf, h, tetol, xi);

for i = 1:1:3
    
    % initial velocity vector of the transfer orbit
    
    vito(i) = vi(i) + x(i);
    
    % predicted state vector of the final mission orbit
    
    rfp(i) = xf(i);
    
    if (otype == 2)
        
        vfp(i) = xf(i + 3) + x(i + 3);
        
    else
        
        vfp(i) = vf(i);
        
    end
    
end

fprintf('\n\n            program lambert4\n');

fprintf('\n< perturbed Earth orbit Lambert problem >\n\n');

fprintf('\norbital elements and state vector of the initial orbit');
fprintf('\n------------------------------------------------------\n');

oeprint1(mu, oev1);

svprint(ri, vi);

fprintf('\norbital elements and state vector of the transfer orbit after the initial delta-v');
fprintf('\n---------------------------------------------------------------------------------\n');

oev = eci2orb1 (mu, ri', vito');

oeprint1(mu, oev);

svprint(ri', vito');

fprintf('\norbital elements and state vector of the transfer orbit prior to the final delta-v');
fprintf('\n----------------------------------------------------------------------------------\n');

oev = eci2orb1 (mu, rfp', (xf(4:6))');

oeprint1(mu, oev);

svprint(rfp', (xf(4:6))');

fprintf('\norbital elements and state vector of the final orbit');
fprintf('\n----------------------------------------------------\n');

oev = eci2orb1 (mu, rfp', vfp');

oeprint1(mu, oev);

svprint(rfp', vfp');

fprintf('\ninitial delta-v vector and magnitude');
fprintf('\n------------------------------------\n');

fprintf('\nx-component of delta-v      %12.6f  meters/second', 1000.0 * x(1));

fprintf('\ny-component of delta-v      %12.6f  meters/second', 1000.0 * x(2));

fprintf('\nz-component of delta-v      %12.6f  meters/second', 1000.0 * x(3));

fprintf('\n\ndelta-v magnitude           %12.6f  meters/second\n', 1000.0 * norm(x(1:3)));

if (otype == 2)
    
    fprintf('\nfinal delta-v vector and magnitude');
    fprintf('\n----------------------------------\n');
    
    fprintf('\nx-component of delta-v      %12.6f  meters/second', 1000.0 * x(4));
    
    fprintf('\ny-component of delta-v      %12.6f  meters/second', 1000.0 * x(5));
    
    fprintf('\nz-component of delta-v      %12.6f  meters/second', 1000.0 * x(6));
    
    fprintf('\n\ndelta-v magnitude           %12.6f  meters/second\n', 1000.0 * norm(x(4:6)));
    
    fprintf('\n\ntotal delta-v               %12.6f  meters/second\n', ...
        1000.0 * (norm(x(1:3)) + norm(x(4:6))));
    
end

fprintf('\n\nfinal position vector error components and magnitude');
fprintf('\n----------------------------------------------------\n');

fprintf('\nx-component of delta-r      %12.8f  meters', 1000.0 * drf(1));

fprintf('\ny-component of delta-r      %12.8f  meters', 1000.0 * drf(2));

fprintf('\nz-component of delta-r      %12.8f  meters', 1000.0 * drf(3));

fprintf('\n\ndelta-r magnitude           %12.8f  meters\n', 1000.0 * norm(drf));

if (otype == 2)
    
    % rendezvous
    
    fprintf('\nfinal velocity vector error components and magnitude');
    fprintf('\n----------------------------------------------------\n');
    
    fprintf('\nx-component of delta-v      %12.8f  meters/second', 1000.0 * dvf(1));
    
    fprintf('\ny-component of delta-v      %12.8f  meters/second', 1000.0 * dvf(2));
    
    fprintf('\nz-component of delta-v      %12.8f  meters/second', 1000.0 * dvf(3));
    
    fprintf('\n\ndelta-v magnitude           %12.8f  meters/second\n\n', 1000.0 * norm(dvf));
    
end

fprintf('\ntransfer time               %12.6f  minutes\n\n', ttmins);
