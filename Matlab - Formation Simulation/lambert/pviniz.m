function pviniz (tof, r1, v1, dv1, dv2)

% primer vector initialization

% required by primer.m

% Orbital Mechanics with Matlab

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global mu pvi pvdi

% compute primer vector at first impulse
  
dv1m = norm(dv1);

pvi = dv1 / dv1m;

% compute primer vector at second impulse
  
dv2m = norm(dv2);

pvf = dv2 / dv2m;
 
% compute state transition matrix

[r2, v2, stm] = stm2(mu, tof, r1, v1);

% extract submatrices of state transition matrix

stm11(1:3, 1:3) = stm(1:3, 1:3);

stm12(1:3, 1:3) = stm(1:3, 4:6);

% compute initial value of primer derivative vector

pvdi = inv(stm12) * (pvf' - stm11 * pvi');

% compute final value of primer derivative vector

stm21(1:3, 1:3) = stm(4:6, 1:3);

stm22(1:3, 1:3) = stm(4:6, 4:6);

pvdf = stm21 * pvi' + stm22 * inv(stm12) * (pvf' - stm11 * pvi');



