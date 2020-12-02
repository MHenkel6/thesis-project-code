% lambert4.m     December 16, 2012

% NLP solution of the perturbed Earth orbit Lambert problem

% flyby and rendezvous trajectory types

% Orbital Mechanics with Matlab

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v1,v2] = lambertJ2(R1,R2,V1,V2, t)

global mu j2 req neq tetol rkcoef

mu = Constants.muEarth;
j2 = Constants.J2;
global otype ri vi tof rtarget vtarget drf dvf

% initialize rkf78 algorithm

rkcoef = 1;

% rkf78 convergence tolerance

tetol = 1.0e-8;

% number of differential equations

neq = 6;


otype = 1;
% time of flight (seconds)

tof = t;

revmax = 0;

% compute state vectors of initial and final orbits
ri = R1;
vi = V1;
rf = R2;
vf = V2;
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

% solve the orbital TPBVP using SNOPT


[x, f, inform, xmul, fmul] = snopt(xg, xlwr, xupr, flow, fupp, 'tpbvp');

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
    
    v1(i) = vi(i) + x(i);
    
    % predicted state vector of the final mission orbit
    
    rfp(i) = xf(i);
    
    if (otype == 2)
        
       vfp(i) = xf(i + 3) + x(i + 3);
       
    else
        
       vfp(i) = vf(i);
       
    end
    
end
v1 = vi + x(1:3);
v2 = xf(4:6);
end