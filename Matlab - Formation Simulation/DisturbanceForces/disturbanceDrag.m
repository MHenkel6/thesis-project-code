function distForce = disturbanceDrag(posECI,velECI, time)
%DISTURBANCEDRAG Determine disturbance force from aerodynamic drag

lat = asin(posECI(3)/norm(posECI));
velAtmo = Constants.planetRot*rotZ(-pi/2)*(posECI'- dot(posECI',[0,0,1]')*[0,0,1]'); %atmospheric velocity, adjust later
velRel = velECI - velAtmo';% Velocity relative to atmosphere
A = 3*sqrt(3)/2; 
m = 100;
Bc = 3 * A / m; % Ballistic coefficient
rho = 1E-12; % atmospheric densit

distForce =  -0.5*Bc*rho*velRel*norm(velRel);

end

