% Analytically calculate jacobian matrix
syms x y z xdot ydot zdot r n ndot mu
jacobian([xdot;
          ydot;
          zdot;
          mu/r^2 - mu*(r+x)/((r+x)^2 + y^2 + z^2)^(3/2) + 2*n*ydot + ndot*y + n^2*x;
                 - mu*y/((r+x)^2 + y^2 + z^2)^(3/2) - 2*n*xdot - ndot*x + n^2*y;
                 - mu*z/((r+x)^2 + y^2 + z^2)^(3/2)],[x,y,z,xdot,ydot,zdot])

