% wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
function [V1, V2] = lambertBook(R1, R2, t, string)
% wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
%{
This function solves Lambert's problem.
mu - gravitational parameter (km^3/s^2)
R1, R2 - initial and final position vectors (km)
r1, r2 - magnitudes of R1 and R2
t - the time of flight from R1 to R2 (a constant) (s)
V1, V2 - initial and final velocity vectors (km/s)
c12 - cross product of R1 into R2
theta - angle between R1 and R2
string - 'pro' if the orbit is prograde
e68 MATLAB Scripts
'retro' if the orbit is retrograde
A - a constant given by Equation 5.35
z - alpha*x^2, where alpha is the reciprocal of the
semimajor axis and x is the universal anomaly
y(z) - a function of z given by Equation 5.38
F(z,t) - a function of the variable z and constant t,
- given by Equation 5.40
dFdz(z) - the derivative of F(z,t), given by Equation 5.43
ratio - F/dFdz
tol - tolerance on precision of convergence
nmax - maximum number of iterations of Newton's procedure
f, g - Lagrange coefficients
gdot - time derivative of g
stumpC(z), stumpS(z) - Stumpff functions
dum - a dummy variable
User M-functions required: stumpC and stumpS
%}
% ----------------------------------------------
mu = Constants.muEarth;
%...Magnitudes of R1 and R2:
r1 = norm(R1);
r2 = norm(R2);
c12 = cross(R1, R2);
theta = angleVec(R1,R2);
%...Determine whether the orbit is prograde or retrograde:
if nargin < 4 || (~strcmp(string,'retro') & (~strcmp(string,'pro')))
    string = 'pro';
    fprintf('\n ** Prograde trajectory assumed.\n')
end
if strcmp(string,'pro')
    if c12(3) <= 0
        theta = 2*pi - theta;
    end
elseif strcmp(string,'retro')
    if c12(3) >= 0
        theta = 2*pi - theta;
    end
end
%...Equation 5.35:
A = sin(theta)*sqrt(r1*r2/(1 - cos(theta)));
%...Determine approximately where F(z,t) changes sign, and
%...use that value of z as the starting value for Equation 5.45:
z = -100;
while F(z,t) < 0
    z = z + 0.1;
end
%...Set an error tolerance and a limit on the number of iterations:
tol = 1.e-8;
nmax = 5000;
%...Iterate on Equation 5.45 until z is determined to within the
%...error tolerance:
ratio = 1;
n = 0;
while (abs(ratio) > tol) && (n <= nmax)
    n = n + 1;
    ratio = F(z,t)/dFdz(z);
    z = z - ratio;
end
%...Report if the maximum number of iterations is exceeded:
if n >= nmax
    fprintf('\n\n **Number of iterations exceeds %g \n\n ',nmax)
end
%...Equation 5.46a:
f = 1 - y(z)/r1;
%...Equation 5.46b:
g = A*sqrt(y(z)/mu);
%...Equation 5.46d:
gdot = 1 - y(z)/r2;
%...Equation 5.28:
V1 = real(1/g*(R2 - f*R1));
%...Equation 5.29:
V2 = real(1/g*(gdot*R2 - R1));
return
% wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
% Subfunctions used in the main body:
% wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
%...Equation 5.38:
    function dum = y(z)
        dum = r1 + r2 + A*(z*stumpS(z) - 1)/sqrt(stumpC(z));
    end
%...Equation 5.40:
    function dum = F(z,t)
        dum = (y(z)/stumpC(z))^1.5*stumpS(z) + A*sqrt(y(z)) - sqrt(mu)*t;
    end
%...Equation 5.43:
    function dum = dFdz(z)
        if z == 0
            dum = sqrt(2)/40*y(0)^1.5 + A/8*(sqrt(y(0)) + A*sqrt(1/2/y(0)));
        else
            C = stumpC(z);
            S = stumpS(z);
            dum = (y(z)/C)^1.5*(1/2/z*(C - 3*S/2/C) + 3*S^2/4/C) + ...
                A/8*(3*S/C*sqrt(y(z)) + A*sqrt(C/y(z)));
        end
    end

end %lambert