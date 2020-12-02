function relativeMeasurement = relativeMeasNonlinear(formationState)


stateSquare = reshape(formationState,[6,6])';
stateCenterECI = mean(stateSquare);
posCenterECI = stateCenterECI(1:3);
velCenterECI = stateCenterECI(4:6);

[a,~,inc,O,~,theta,~,argLat,~] = rv2orb(posCenterECI',velCenterECI');
n = sqrt(Constants.muEarth/a^3);
%argLat = argLat + pi/2;
theta = real(theta);

% Transform into Hill Frame
x = velCenterECI/norm(velCenterECI);
z = posCenterECI/norm(posCenterECI);
y = cross(z,x)/(norm(cross(z,x)));
THillECI = [x;
            y;
            z];
relState = stateSquare-repmat(stateCenterECI,6,1);
Tdot = n*rotX(0.5*pi)*rotZdot(argLat)*rotZ(0.5*pi)*rotX(inc)*rotZ(O);
posHill = THillECI*(relState(:,1:3))';
vHill = THillECI*(relState(:,4:6))'-THillECI*Tdot*THillECI*(relState(:,1:3))';
stateHill = [posHill;vHill]';

relativeMeasurement = zeros(1,144);
% Satellite 1 Neighbros
relativeMeasurement(1:6) = stateHill(2,:)-stateHill(1,:);
relativeMeasurement(7:12) = stateHill(3,:)-stateHill(1,:);
relativeMeasurement(13:18) = stateHill(4,:)-stateHill(1,:);
relativeMeasurement(19:24) = stateHill(5,:)-stateHill(1,:);

% Satellite 2 Neighbros
relativeMeasurement(25:30) = stateHill(1,:)-stateHill(2,:);
relativeMeasurement(31:36) = stateHill(5,:)-stateHill(2,:);
relativeMeasurement(37:42) = stateHill(6,:)-stateHill(2,:);
relativeMeasurement(43:48) = stateHill(3,:)-stateHill(2,:);

% Satellite 3 Neighbros
relativeMeasurement(49:54) = stateHill(1,:)-stateHill(3,:);
relativeMeasurement(55:60) = stateHill(2,:)-stateHill(3,:);
relativeMeasurement(61:66) = stateHill(6,:)-stateHill(3,:);
relativeMeasurement(67:72) = stateHill(4,:)-stateHill(3,:);

% Satellite 4 Neighbros
relativeMeasurement(73:78) = stateHill(3,:)-stateHill(4,:);
relativeMeasurement(79:84) = stateHill(6,:)-stateHill(4,:);
relativeMeasurement(85:90) = stateHill(5,:)-stateHill(4,:);
relativeMeasurement(91:96) = stateHill(1,:)-stateHill(4,:);

% Satellite 5 Neighbros
relativeMeasurement(97:102) = stateHill(4,:)-stateHill(5,:);
relativeMeasurement(103:108) = stateHill(6,:)-stateHill(5,:);
relativeMeasurement(109:114) = stateHill(2,:)-stateHill(5,:);
relativeMeasurement(115:120) = stateHill(1,:)-stateHill(5,:);

% Satellite 6 Neighbros
relativeMeasurement(121:126) = stateHill(5,:)-stateHill(6,:);
relativeMeasurement(127:132) = stateHill(4,:)-stateHill(6,:);
relativeMeasurement(133:138) = stateHill(3,:)-stateHill(6,:);
relativeMeasurement(139:144) = stateHill(2,:)-stateHill(6,:);


end

