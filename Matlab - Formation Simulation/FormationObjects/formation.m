classdef formation < handle
    %FORMATION Class containing the formation as a whole and relevant
    %methods
    properties
        spacecraftArray; % Array containing spacecraft objects in formation
        nSpacecraft; % no. of spacecraft
        formationSize;  % distance between satellite
        nImpulse; % number of impulses used to control orbit
        
        % Center Orbit parameters
        orbitParam; % Array to jointly hold orbital parameters
        a; % Semi-major axis [m] 
        e; % Eccentricity [-]
        inc; % Inclination [rad]
        O; % RAAN [rad]
        o; % Argument of periapsis [rad]
        nu; % True anomaly [rad]
        truLon; % True Longitude [rad]
        argLat; % Argument of latitude [rad]
        lonPer; % Longitude of Periapsis [rad]
        p; % Semi-latus rectum [m] 
        
        % Control parameteers
        controltype; % Continuous (1) vs discrete (2) control system 
        disctype; % Discretization type if discrete control system
        
        % Navigation parameters
        navType = 1;  % Cartesian (1) or Spharical (2) noise and bias
        
        % Propagation parameters
        disturbancesOn = 1; % Boolean to en/disable disturbanceForces
    end
    
    methods
        function obj = formation(nSpacecraft,type,center,velCenter,dist,qOrientation,controltype,disctype,spacecraftParameters,residualCovariance,residualMean,navType,navArray)
            %FORMATION Construct an instance of this class
            %   Inputs:
            %   nSpacecraft             = number of spacecraft in formation
            %   type                    = Formation type
            %   center                  = position of geometric center of Formation in ECI
            %   velCenter               = velocity of geometric center of Formation in ECI
            %   dist                    = size parameter of formation (distance between
            %                             sats)
            %   qOrientation            = orientation quaternion relative to ECI
            %   controlType             = continuous vs discrete control
            %                             method
            %   discType                = discretization type
            %                             (ZOH,FOH,Impulse)
            %   spacecraftParameters    = array of mass, size, moment of
            %                             inertia and other relevant 
            %                             spacecraft parameters, see
            %                             spacracraft class
            %   residualCovariance      = Covariance matrix of residual
            %                             vector in the faultless case 
            %                             (determined from the
            %                             simulation)
            %   residualMean            = mean of the residual vector
            
            % Construct 6 spacecraft objects  
            obj.nSpacecraft = nSpacecraft;
            obj.formationSize = dist;
            % If octahedron type. get 6 equidistant points from center
            switch type
                case 'octahedron'
                    distCenter = dist*sqrt(2)/2; %distance from center
                    distArray = [0,0,distCenter;
                                 distCenter,0,0;
                                 0,distCenter,0;
                                 -distCenter,0,0;
                                 0 -distCenter,0;
                                 0,0,-distCenter];
                case 'single'
                    distArray = [0,0,0];
                otherwise
                    warning('Unsupported Formation Type')
            end
            % Reorient positions to align with formation frame
            distArrayRot =  quatrotate(qOrientation,distArray);
            positions = center + distArrayRot+1*randn(6,3);         
                     
            % Reference Orbit
            [a,e,inc,O,o,nu,truLon,argLat,lonPer,p] = rv2orb(center',velCenter');
            obj.a = a;
            obj.e = e;
            obj.inc = inc;
            obj.O = O;
            obj.o = o;
            obj.nu = nu;
            obj.truLon = truLon; 
            obj.argLat = argLat;
            obj.lonPer = lonPer;
            obj.p = p;
            obj.orbitParam = [a,e,inc,O,o,nu,truLon,argLat,lonPer,p];
            % Create Spacecraft
            sats=[];
            for inc = 1:nSpacecraft
                 % Fixed attitude
                attQuat = qOrientation';
                spinVec = zeros([3,1]);
                spinRate = 0;
                formationParameters = [inc,dist,qOrientation,obj.orbitParam];
                sp  = spacecraft(positions(inc,:),velCenter,attQuat,spinRate*spinVec,...
                                 spacecraftParameters,formationParameters);
                sats = [sats(:)',sp];
            end
            
            
            % Give each spacecraft its neighbors
            % Neighbor sat 1
            sats(1).spacecraftNeighbor1 = sats(2);
            sats(1).spacecraftNeighbor2 = sats(3);
            sats(1).spacecraftNeighbor3 = sats(4);
            sats(1).spacecraftNeighbor4 = sats(5);
            
            sats(1).n1Offset = -(sats(2).centerOffset - sats(1).centerOffset);
            sats(1).n2Offset = -(sats(3).centerOffset - sats(1).centerOffset);
            sats(1).n3Offset = -(sats(4).centerOffset - sats(1).centerOffset);
            sats(1).n4Offset = -(sats(5).centerOffset - sats(1).centerOffset);
            % Neighbor sat 2
            sats(2).spacecraftNeighbor1 = sats(1);
            sats(2).spacecraftNeighbor2 = sats(5);
            sats(2).spacecraftNeighbor3 = sats(6);
            sats(2).spacecraftNeighbor4 = sats(3);
            
            sats(2).n1Offset = -(sats(1).centerOffset - sats(2).centerOffset);
            sats(2).n2Offset = -(sats(5).centerOffset - sats(2).centerOffset);
            sats(2).n3Offset = -(sats(6).centerOffset - sats(2).centerOffset);
            sats(2).n4Offset = -(sats(3).centerOffset - sats(2).centerOffset);
            % Neighbor sat 3
            sats(3).spacecraftNeighbor1 = sats(1);
            sats(3).spacecraftNeighbor2 = sats(2);
            sats(3).spacecraftNeighbor3 = sats(6);
            sats(3).spacecraftNeighbor4 = sats(4);
            
            sats(3).n1Offset = -(sats(1).centerOffset - sats(3).centerOffset);
            sats(3).n2Offset = -(sats(2).centerOffset - sats(3).centerOffset);
            sats(3).n3Offset = -(sats(6).centerOffset - sats(3).centerOffset);
            sats(3).n4Offset = -(sats(4).centerOffset - sats(3).centerOffset);
            % Neighbor sat 4
            sats(4).spacecraftNeighbor1 = sats(3);
            sats(4).spacecraftNeighbor2 = sats(6);
            sats(4).spacecraftNeighbor3 = sats(5);
            sats(4).spacecraftNeighbor4 = sats(1);
            
            sats(4).n1Offset = -(sats(3).centerOffset - sats(4).centerOffset);
            sats(4).n2Offset = -(sats(6).centerOffset - sats(4).centerOffset);
            sats(4).n3Offset = -(sats(5).centerOffset - sats(4).centerOffset);
            sats(4).n4Offset = -(sats(1).centerOffset - sats(4).centerOffset);
            % Neighbor sat 5
            sats(5).spacecraftNeighbor1 = sats(4);
            sats(5).spacecraftNeighbor2 = sats(6);
            sats(5).spacecraftNeighbor3 = sats(2);
            sats(5).spacecraftNeighbor4 = sats(1);
            
            sats(5).n1Offset = -(sats(4).centerOffset - sats(5).centerOffset);
            sats(5).n2Offset = -(sats(6).centerOffset - sats(5).centerOffset);
            sats(5).n3Offset = -(sats(2).centerOffset - sats(5).centerOffset);
            sats(5).n4Offset = -(sats(1).centerOffset - sats(5).centerOffset);
            % Neighbor sat 6
            sats(6).spacecraftNeighbor1 = sats(5);
            sats(6).spacecraftNeighbor2 = sats(4);
            sats(6).spacecraftNeighbor3 = sats(3);
            sats(6).spacecraftNeighbor4 = sats(2);
            
            sats(6).n1Offset = -(sats(5).centerOffset - sats(6).centerOffset);
            sats(6).n2Offset = -(sats(4).centerOffset - sats(6).centerOffset);
            sats(6).n3Offset = -(sats(3).centerOffset - sats(6).centerOffset);
            sats(6).n4Offset = -(sats(2).centerOffset - sats(6).centerOffset);
            % Assign satellite array to formation property
            obj.spacecraftArray = sats;
            
            obj.controltype = controltype;
            obj.disctype = disctype;
            if exist('navType')
                obj.navType = navType;
            else
                obj.navType = 1;
            end
            
            if navType == 2
                rangeNoiseSize = navArray(1);
                rangeBiasSize = navArray(2);
                angleNoiseSize = navArray(3);
                angleBiasSize = navArray(4);
                for sc = obj.spacecraftArray
                    sc.rangeBias = (2*randi(2,4,1)-3)*rangeBiasSize;
                    sc.rangeNoise = rangeNoiseSize;
                    sc.angleBias = (2*randi(2,4,2)-3)*angleBiasSize;
                    sc.angleNoise =  angleNoiseSize;
                end
            end
            % Give spacecraft 1 the residual Covariance for FDI function
            obj.spacecraftArray(1).residualMean = residualMean;
            % Finish off with inital measurements and control commands
            for ii = 1:6
                obj.spacecraftArray(ii).navigation(obj.navType);
                obj.spacecraftArray(ii).controlCommand(0,obj.controltype,obj.disctype);
                obj.spacecraftArray(ii).thrustAlloc(0);
                cF = obj.spacecraftArray(ii).controlForce(0);
            end
        end
        
        function rk4Prop(obj,time,dt)
            %rk4Prop Propagate the states of the formation
            %   Main function to propagate the state of the entire
            %   formation by one time step dt
            
            % nSpacecraft x 6 Matrix containing state derivatives for
            % all spacecraft
            devArray = zeros(obj.nSpacecraft,6); 
            KalmanMeasurement = zeros(6,24); 
            KalmanControlStates = zeros(1,3*6);
            % Determine state changes for each satellite
            for i = 1:obj.nSpacecraft
                spacecraft =  obj.spacecraftArray(i);
                %spacecraft.guidance(time);
                
                % Perform relative and absolute measurements
                spacecraft.navigation(obj.navType);
                % Determine thruster output
                spacecraft.controlCommand(time,obj.controltype,obj.disctype);
                % Determine thruster opening time based on commanded input
                spacecraft.thrustAlloc(time);
                % Determine actual thrust exerted by thrusters in ECI frame
                cF = spacecraft.controlForce(time);
                
                % Gather measurements and control states for the Kalman
                % filter
                KalmanMeasurement(i,:) = reshape(spacecraft.relEstHill',1,24);
                KalmanControlStates(1,(i-1)*3+1:i*3) = spacecraft.cAccHill;                
                
                % Runge-Kutta 4 Integration Scheme
                dev1 = dynamics(time, spacecraft.position,spacecraft.velocity,...
                                spacecraft.mass,cF,obj.disturbancesOn);
                dev2 = dynamics(time+dt/2, spacecraft.position+dt/2*dev1(1:3),...
                                spacecraft.velocity+dt/2*dev1(4:6),...
                                spacecraft.mass,cF,obj.disturbancesOn);
                dev3 = dynamics(time+dt/2, spacecraft.position+dt/2*dev2(1:3),...
                                spacecraft.velocity+dt/2*dev2(4:6),...
                                spacecraft.mass,cF,obj.disturbancesOn);
                dev4 = dynamics(time+dt, spacecraft.position+dt*dev3(1:3),...
                                spacecraft.velocity+dt*dev3(4:6),...
                                spacecraft.mass,cF,obj.disturbancesOn);                   
                devTotal = dt*(dev1 + 2*dev2 + 2*dev3 + dev4)/6;
                devArray(i,:) = devTotal;   
                
                % Determine change in attitude and rotation
                rotState = [spacecraft.attitude;
                            spacecraft.spin];
                rotDev1 = rotDynamics(rotState,spacecraft.inertia);
                rotDev2 = rotDynamics(rotState+rotDev1*dt/2,spacecraft.inertia);
                rotDev3 = rotDynamics(rotState+rotDev2*dt/2,spacecraft.inertia);
                rotDev4 = rotDynamics(rotState+rotDev3*dt,spacecraft.inertia);
                                
                rotDevTotal = dt*(rotDev1 + 2*rotDev2 + 2*rotDev3 + rotDev4)/6;
                spacecraft.updateAtt(rotDevTotal);                
            end
            % Kalman filer update
            % Use all relative Measurements in ECI to get proper Hill frame
            
            % Reorder Measurements to be in correct order
            % Correct Order: [s12,s13,s14,s15,s23,s34,s45,s52,s65,s64,s63,s62, ...
            %                 s21,s31,s41,s51,s32,s43,s54,s25,s56,s46,s36,s26]
            % Current Order:
            %                [s12,s13,s14,s15,s21,s25,s26,s23,s31,s32,s36,s34,...
            %                 s43,s46,s45,s41,s54,s56,s52,s51,s65,s64,s63,s62
            
            KalmanMeasurementCorrect = zeros(1,144);
            KalmanMeasurementCorrect(1:24)    = KalmanMeasurement(1,:);
            KalmanMeasurementCorrect(25:30)   = KalmanMeasurement(2,19:24);
            KalmanMeasurementCorrect(31:36)   = KalmanMeasurement(3,19:24);
            KalmanMeasurementCorrect(37:42)   = KalmanMeasurement(4,13:18);
            KalmanMeasurementCorrect(43:48)   = KalmanMeasurement(5,13:18);
            KalmanMeasurementCorrect(49:72)   = KalmanMeasurement(6,:);
            
            KalmanMeasurementCorrect(73:78)   = KalmanMeasurement(2,1:6);
            KalmanMeasurementCorrect(79:84)   = KalmanMeasurement(3,1:6);
            KalmanMeasurementCorrect(85:90)   = KalmanMeasurement(4,19:24);
            KalmanMeasurementCorrect(91:96)   = KalmanMeasurement(5,19:24);
            KalmanMeasurementCorrect(97:102)  = KalmanMeasurement(3,7:12);
            KalmanMeasurementCorrect(103:108) = KalmanMeasurement(4,1:6);
            KalmanMeasurementCorrect(109:114) = KalmanMeasurement(5,1:6);
            KalmanMeasurementCorrect(115:120) = KalmanMeasurement(2,7:12);
            KalmanMeasurementCorrect(121:126) = KalmanMeasurement(5,7:12);
            KalmanMeasurementCorrect(127:132) = KalmanMeasurement(4,7:12);
            KalmanMeasurementCorrect(133:138) = KalmanMeasurement(3,13:18);
            KalmanMeasurementCorrect(139:144) = KalmanMeasurement(2,13:18);
            
            % Perform Kalman filter update
            obj.spacecraftArray(1).kalmanUpdate(KalmanMeasurementCorrect,KalmanControlStates);
            %obj.spacecraftArray(1).extendedKalmanUpdate(time,KalmanMeasurementCorrect,KalmanControlStates);
            
            % Perform FDI once residual signal has settle (~1000 seconds
            if time>1000
                obj.spacecraftArray(1).fdi(time)
            end
            
            % Update all positions
            for i = 1:obj.nSpacecraft
                obj.spacecraftArray(i).updatePos(devArray(i,:));   
            end
            % For output purposes, another measurement is taken
            for ii = 1:obj.nSpacecraft
                obj.spacecraftArray(ii).navigation(obj.navType);
            end
        end
        
        function states = getStates(obj)
            % Return true position and velocity of the formation in ECI
            states = zeros(obj.nSpacecraft,6);
            for i = 1:obj.nSpacecraft
                states(i,:) = obj.spacecraftArray(i).getState();
            end
        end
        
        function [relState,relStateHill] = getRelStates(obj, zeroed)
            % Return the relative measurements for each member of the
            % formation, both in ECI and in the HILL frame
            relState = zeros(4*obj.nSpacecraft,6);
            relStateHill = zeros(4*obj.nSpacecraft,6);
            for i = 1:6
                sc = obj.spacecraftArray(i);
                relEst = sc.relativeEst;
                
                relDiff = relEst - [sc.n1Offset,0,0,0;
                                    sc.n2Offset,0,0,0;
                                    sc.n3Offset,0,0,0;
                                    sc.n4Offset,0,0,0];
                if zeroed   
                    relState((i-1)*4+1:i*4,:) = relDiff;
                else
                    relState((i-1)*4+1:i*4,:) = relEst;
                end
                relStateHill((i-1)*4+1:i*4,:) = sc.relEstHill;
            end
        end
        function [absMeas] = getAbsoluteMeasurement(obj)
            % Return measured position and velocity of the formation in ECI
            absMeas = zeros(6,6);
            for ii = 1:6
                absMeas(ii,:) = [obj.spacecraftArray(ii).positionEst,...
                                 obj.spacecraftArray(ii).velocityEst];
            end
        end
        function [command,err] = getControlCommands(obj)
            % Return current commanded control acceleration 
            command = zeros(6,3);
            err = zeros(6,6);
            for ii = 1:6
                sc = obj.spacecraftArray(ii);
                command(ii,:) = sc.cAccHill;% sc.cImpulseHill;%
                err(ii,:) = sc.cErr; % control err;
            end
            
        end
        function setFault(obj,faultTime,satelliteNo,thrusterNo,faultType,faultParam)
            % Set faultvector of selected satellite and thruster to 
            % faultparameter
            faultySat = obj.spacecraftArray(satelliteNo);
            faultySat.faultTime = faultTime;
           
            if faultType == 1 % Closed thruster fault 
                faultySat.faultVectorClosed(thrusterNo) = 1-faultParam;
            elseif faultType == 2 % Open thruster
                faultySat.faultVectorOpen(thrusterNo) = faultParam;
            end
        end

        %% Following functions mainly used in the optimization of the formation
        function setStates(obj,center,size,qOrientation,velCenter)
            % Set state of formation for optimization
            distCenter = size*sqrt(2)/2; %distance from center
            distArray = [0,0,distCenter;
                         distCenter,0,0;
                         0,distCenter,0;
                         -distCenter,0,0;
                         0 -distCenter,0;
                         0,0,-distCenter];
            distArrayRot =  quatrotate(qOrientation,distArray);
            positions = center + distArrayRot; 
            it = 1 ; 
            for sat = obj.spacecraftArray
                sat.setState([positions(it,:),velCenter]);
            end
        end
                
%         function [deltaVs, positions] = guideOpt(obj,dt,T)
%             % Calculate deltaV required to keep current formation for 1
%             % orbit 
%             % Returns required DeltaVs per satellite and the positions wrt
%             % the center of the formation.            
%             
%             % Propagate orbit for one revolution with Keplerian dynamics
%             % only
%             for t = 0:dt:T
%                 devArray = zeros(obj.nSpacecraft,6); 
%                 for i = 1:obj.nSpacecraft
%                     spacecraft =  obj.spacecraftArray(i);
%                     dev1 = dynamicsKepler(spacecraft.position,spacecraft.velocity);
%                     dev2 = dynamicsKepler(spacecraft.position+dt/2*dev1(1:3),...
%                                           spacecraft.velocity+dt/2*dev1(4:6));                                      
%                     dev3 = dynamicsKepler(spacecraft.position+dt/2*dev2(1:3),...
%                                           spacecraft.velocity+dt/2*dev2(4:6));                                      
%                     dev4 = dynamicsKepler(spacecraft.position+dt*dev3(1:3),...
%                                           spacecraft.velocity+dt*dev3(4:6));
% 
%                     devTotal = dt*(dev1 + 2*dev2 + 2*dev3 + dev4)/6;
%                     devArray(i,:) = devTotal;   
%                 end
%                 % Update all positions
%                 for i = 1:obj.nSpacecraft
%                     obj.spacecraftArray(i).updatePos(devArray(i,:));
%                 end
%             end
%             deltaVs = 0;
%             positions = 0;
%         end
            
    end
end

