close all
clc
%% Script to verify LQR Controller 
% Discrete time system parameters
a = 0.5;
b = 1;
q = 2;
r = 0.5;
A = [a];
B = [b];
Q = [q];
R = [r];
E = [1];
% Solve Discrete Algrbraic Ricatti Equation using idare()
[S,~,~,info] = idare(A,B,Q,R,0,E);
% Compute LQR gain
K =(R+transpose(B)*S*B)\transpose(B)*S*A;
astar = (-b^2);
bstar = (a^2*r + b^2*q - r);
cstar = q*r;
s = max((-bstar-sqrt(bstar^2-4*astar*cstar))/(2*astar),(-bstar+sqrt(bstar^2-4*astar*cstar))/(2*astar))
k = a*b*s/(b^2*s+r)
% Differences between analytical and numerical
sdiff = s-S
kdiff = k-K
