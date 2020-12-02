function J = dynJac(formationState,formCenter,velCenter,a,e,argLat)
% Return Jacobian matrix
J = [];
r = norm(formCenter);
mu = Constants.muEarth;
n = sqrt(Constants.muEarth/(a^3*(1-e^2)^3))*(1+e*cos(argLat))^2;
rdot = norm(dot(formCenter,velCenter)/norm(formCenter));
ndot = -2*rdot*n/r;
for ii = 1:12
    x  = formationState((ii-1)*6+1);
    y  = formationState((ii-1)*6+2);
    z  = formationState((ii-1)*6+3);
    
    A = [                                                                                                     0,                                                                                 0,                                                                           0,    1,   0, 0;
                                                                                                              0,                                                                                 0,                                                                           0,    0,   1, 0;
                                                                                                              0,                                                                                 0,                                                                           0,    0,   0, 1;
          n^2 - mu/((r + x)^2 + y^2 + z^2)^(3/2) + (3*mu*(2*r + 2*x)*(r + x))/(2*((r + x)^2 + y^2 + z^2)^(5/2)),                             ndot + (3*mu*y*(r + x))/((r + x)^2 + y^2 + z^2)^(5/2),                              (3*mu*z*(r + x))/((r + x)^2 + y^2 + z^2)^(5/2),    0, 2*n, 0;
                                                  (3*mu*y*(2*r + 2*x))/(2*((r + x)^2 + y^2 + z^2)^(5/2)) - ndot, n^2 - mu/((r + x)^2 + y^2 + z^2)^(3/2) + (3*mu*y^2)/((r + x)^2 + y^2 + z^2)^(5/2),                                    (3*mu*y*z)/((r + x)^2 + y^2 + z^2)^(5/2), -2*n,   0, 0;
                                                         (3*mu*z*(2*r + 2*x))/(2*((r + x)^2 + y^2 + z^2)^(5/2)),                                          (3*mu*y*z)/((r + x)^2 + y^2 + z^2)^(5/2), (3*mu*z^2)/((r + x)^2 + y^2 + z^2)^(5/2) - mu/((r + x)^2 + y^2 + z^2)^(3/2),    0,   0, 0];
    J = blkdiag(J,A);
end

end

