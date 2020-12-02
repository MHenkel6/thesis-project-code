function [f, g] = tpbvp(x)

% two point boundary value objective function
% and state vector constraints

% input

%  x = current delta-v vector

% output

%  f(1) = objective function (delta-v magnitude)
%  f(2) = rx constraint delta
%  f(3) = ry constraint delta
%  f(4) = rz constraint delta
%  f(5) = vx constraint delta
%  f(6) = vy constraint delta
%  f(7) = vz constraint delta

% Orbital Mechanics with Matlab

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global otype neq tetol

global ri vi tof rtarget vtarget drf dvf

% load current state vector of transfer orbit
  
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

% objective function (delta-v magnitude)

if (otype == 1)
   % initial delta-v only (flyby)
   
   f(1) = norm(x);
else
   % total delta-v (rendezvous)
   
   f(1) = norm(x(1:3)) + norm(x(4:6));    
end

% final position vector equality constraints
    
f(2) = rtarget(1) - xf(1);

f(3) = rtarget(2) - xf(2);

f(4) = rtarget(3) - xf(3);

if (otype == 2)
   % final velocity vector

   vf(1) = xf(4) + x(4);
   vf(2) = xf(5) + x(5);
   vf(3) = xf(6) + x(6);
end

if (otype == 2)
   % enforce final velocity vector constraints

   f(5) = vtarget(1) - vf(1);

   f(6) = vtarget(2) - vf(2);

   f(7) = vtarget(3) - vf(3); 
end

% save state vector deltas for print summary

for i = 1:1:3
    drf(i) = f(i + 1);
    
    if (otype == 2)
       % rendezvous
       
       dvf(i) = f(i + 4);
    end
end

% transpose objective function/constraints vector

f = f';

% no derivatives

g = [];
