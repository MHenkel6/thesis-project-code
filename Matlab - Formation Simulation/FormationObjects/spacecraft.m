classdef spacecraft < handle
    properties
        position; % Position in ECI in m, 3 doubles
        velocity; % velocity in ECI in m/s 3 doubles
        attitude; % attitude quaternion in ECI
        spin; % rotation rate in rad/s
        
        positionEst; % estimated position in ECI in m, 3 doubles
        velocityEst; % estimated velocity in ECI in m, 3 doubles
        attitudeEst; % estimated attitude quaternion in ECI 
        
        relativeEst; % estimated/measured relative positions [m] 
        relEstHill;  % Estimated distance,velocity in Hill frame [m, m/s] 
        
        posSelfBias; % position bias for absolute position determination in m
        velSelfBias; % velocity bias for absolute velocity determination in m
        posRelBias; % position bias for relative position determination in m
        velRelBias; % position bias for relative velocity determination in m
        
        posSelfNoise; % standard deviation of absolute position noise [m] 
        velSelfNoise; % standard deviation of absolute velocity noise [m/s]
        posRelNoise; % standard deviation of relative position noise [m] 
        velRelNoise; % standard deviation of relative velocity noise [m/s]
        
        rangeBias; % Bias in range measurement for robustness cases [m]
        angleBias; % Bias in angle measurement for robustness cases [rad]
        
        rangeNoise; % standard deviation in range measurement for robustness cases [m]
        angleNoise; % standard deviation in angle measurement for robustness cases [rad] 
        
        mass = 100 ; % satellite mass [kg]
        inertia = [100,0,0;
                   0,100,0;
                   0,0,100]; % Inertia Tensor
        dim = 1;% satellite dimensions in m
        
        
        % Formation Info
        centerOffset; % Distance to center of ideal formation [m]
        formationNo; % Satellite number in formation 
        formationOrientation; % Orientation quaternion of the formation
        formationSize; % Distance between adjacent satellites [m] 
        formationCenterOrbit; % Orbital parameters of center reference orbit
        
        %Thruster Parameters
        minImpulse = 0.0001; % minimum inpulse bit 
        isp = 200; % specific impulse in s
        thrust = 4; % Maximum thrust force of satellite thruster [4]
        Tconfig = [1,-1,0,0 ,0,0 ;
                   0,0 ,1,-1,0,0 ;
                   0,0 ,0,0 ,1,-1];% thruster configuration matrix
        faultVectorClosed = ones(6,1); 
        faultVectorOpen = zeros(6,1);
        % boolean to adjust fault time to after first firing of affected thruster
        firstThrust = false; 
        % Occurenf of fault time
        faultTime = 1e9;
        
        % Pseudo-inverse of the thruster configuration
        TconfigInv = [0.5,0   ,0   ;
                     -0.5,0   ,0   ;
                      0  ,0.5 ,0   ;
                      0  ,-0.5,0   ;
                      0  ,0   ,0.5 ;
                      0  ,0   ,-0.5]
        
                  
        thrustInterval; % Time in between beginnings of burn windows [s]
        burnTime; % Duration of burn window [s]
        accumDV = 0; % accumulated DV over orbit [m/s]
        maxBurnTime = 0; % maximum obverved burn time;
        thrusterOpeningCount = zeros(6,1); % Amount of times each thruster has opened
        thrusterOpeningTime = zeros(6,1); % Cumulative time of each thruster opening [s]
        spentProp = 0; % Amount of fuel expelled during burns [kg]
        % Satellite neighbors (spacecraft object and offset in ideal
        % conditions)
        spacecraftNeighbor1;
        n1Offset;
        spacecraftNeighbor2;
        n2Offset;
        spacecraftNeighbor3;
        n3Offset;
        spacecraftNeighbor4;
        n4Offset;
        
        %Navigation Matrices
        THillECI; % Transformation matrix from Hill to ECI frame
        Tdot; % Derivative of THillECI
        
        % Controller Parameters      
        Kcont; % Continuous LQR controller gain
        KZOH; % Discrete LQR controller gain using Zero-Order-Hold Discretization
        Kimp; % Discrete LQR controller gain using Impulse-Method Discretization
        
        % Current Control Commands
        cImpulse; % Commanded control impulse in ECI
        cImpulseHill; % Commanded control impulse in Hill Frame
        cFECI; % Actual force output from thruster in ECI [N]
        cAccHill; % Commanded acceleration in Hill Frame [m/s^2]
        cErr; % Control error [m,m/s]
        opTimes; % Opening time vector for thruster allocation [s]
        thrustAllocComplete = 0; % Boolean to indicate thruster allocation
        burnFraction; % vector to scale down thrust 
        pastControl;
        % Kalman filter Matrices 
        
        A; % State matrix
        B; % Input Matrix
        C; % Measurement Matrix
        D; % Feed-through Matrix
        
        Q; % Covariance matrix of system noise
        
        R; % Covariance matrix of measurement noise
        Kk; % Kalman gain
        P;% Solution to Discrete Ricatti Equation from Kalman filter
        M; % Innovation gain
        
        kalmanEstimate; % Current Kalman estimate of formation state
        kalmanResidual; % Current Kalman residual
        ErrCov; % Error Covariance Matrix
        alphLow = 0.025; % Low pass filter parameter
        kalLowPass; % Kalman filter passed through discrete low pass filter
        
        
        % Extended Kalman filter Matrices
        Bf;
        extQ; 
        extKalmanEstimate;
        extKalmanResidual;
        extErrCov;
        
        % Relative state and absolute state from last time step
        lastRelativeEst; 
        lastPositionEst;
        lastVelocityEst;
        
        % Fault Detection  Variables
        residualCovariance; % Covariance Matrix of residual/innovation 
                            % vector in the faultless case
        Qinv; % Inverse of residual Covariance, saved here for increased speed
        residualMean; % Mean of residual vector
        faultThreshold; % Threshold before fault alarm is issued
        gk; % Cumulative sum of Log-likelihood ratio, recursively computed
        detect; % Boolean to indicate detection
        isolate; % Index of detected faulty thruster
        isoCounter; % Counter to indicate successive identifications 
        faultDetTime; % time at which fault was detected [s]
        faultDetected; % Boolean to indicate if fault was found at all
        % FDI matrix
        tempMatrix; % Temporary matrix saved here for reduced computing effort
    end
    
    methods
        function obj = spacecraft(position,velocity, attitude,spin,...
                                  spacecraftParameters,formationParameters)
            %Constructor
            obj.position = position;
            obj.velocity = velocity;
            obj.attitude = attitude;
            obj.spin = spin;
            
            
            % Unpack and assign parameters 
            obj.mass = spacecraftParameters{1};
            obj.dim = spacecraftParameters{2};
            obj.inertia = spacecraftParameters{3};
            obj.thrustInterval = spacecraftParameters{4};
            obj.burnTime =  spacecraftParameters{5};
            obj.thrust = spacecraftParameters{6};
            obj.isp = spacecraftParameters{7};
            obj.minImpulse = 1e-3*spacecraftParameters{8};
            selfBiasSize = spacecraftParameters{9};
            relBiasSize  = spacecraftParameters{10};
            velBiasSize  = spacecraftParameters{11};
            obj.posSelfNoise = spacecraftParameters{12};
            obj.velSelfNoise = spacecraftParameters{14};
            obj.posRelNoise = spacecraftParameters{13};
            obj.velRelNoise = spacecraftParameters{14};
            alpha1 = spacecraftParameters{15};
            alpha2 = spacecraftParameters{16};
            
            % Bias vectors
            obj.posSelfBias = random_unit_vector()'*selfBiasSize;
            obj.velSelfBias = random_unit_vector()'*velBiasSize;
            obj.posRelBias  = random_unit_vector()'*relBiasSize;
            obj.velRelBias  = random_unit_vector()'*velBiasSize;
            
            % Range Navigation biases and noises
            %obj.rangeBias = (2*(randi(2)-1)-1)*Constants.rangeBiasSize;
            %obj.angleBias = [cos(2*pi*rand()),sin(2*pi*rand())]*Constants.angleBiasSize;
            
            % Unpack and assign formation Parameters
            formationNo = formationParameters(1);
            formationSize = formationParameters(2);
            formationOrientation = formationParameters(3:6);
            obj.formationNo = formationNo;
            obj.formationSize = formationSize;
            obj.formationOrientation = formationParameters(3:6);
            obj.formationCenterOrbit = formationParameters(7:16);
            switch formationNo
                case 1
                    offset = -[0,0,formationSize*sqrt(2)/2];                                 
                case 2
                    offset = -[formationSize*sqrt(2)/2,0,0];              
                case 3
                    offset = -[0,formationSize*sqrt(2)/2,0];
                case 4
                    offset = -[ -formationSize*sqrt(2)/2,0,0];
                case 5
                    offset = -[0 -formationSize*sqrt(2)/2,0];
                case 6
                    offset = -[0,0,-formationSize*sqrt(2)/2];
            end
            obj.centerOffset = quatrotate(formationOrientation,offset);
            
            a = obj.formationCenterOrbit(1);
            n = sqrt(Constants.muEarth/a^3);
            
            % Inclusion of J2 effect
            J2 = Constants.J2;
            r = a;
            inc = obj.formationCenterOrbit(3);
            s = (3*J2*Constants.graviPar.Re^2 / (8*r^2)) * (1 + 3*cos(2*inc));
            c = sqrt(1 + s);
            %Test without j2 compensation
            %c = 1;
            % Determine Continuous LQR gain
            A = [0           ,0,0             ,1     ,0    ,0;
                 0           ,0,0             ,0     ,1    ,0;
                 0           ,0,0             ,0     ,0    ,1;
                (5*c^2-2)*n^2,0,0             ,0     ,2*n*c,0;
                 0           ,0,0             ,-2*n*c,0    ,0;
                 0           ,0,-(3*c^2-2)*n^2,0     ,0    ,0]; 
             

            B = [0,0,0;
                 0,0,0;
                 0,0,0;
                 1,0,0;
                 0,1,0;
                 0,0,1];

            contSys = ss(A,B,eye(6,6),0);
            Qcont = zeros(6);
            Qcont(1:3,1:3) = alpha1 * eye(3);
            Qcont(4:6,4:6) = alpha2 * eye(3);
            Qdisc = zeros(6);
            Qdisc(1:3,1:3) = alpha1 * eye(3);
            Qdisc(4:6,4:6) = alpha2 * eye(3);
            %{
            Q(1,1) = 0.5;
            Q(2,2) = 0.5;
            Q(3,3) = 0.5;
            
            Q(4,4) = 1;
            Q(5,5) = 1;
            Q(6,6) = 1;
            %}
            Rcont = eye(3,3);
            Rdisc = 1*eye(3,3);
            E = eye(6,6);
            [S,~,~,~] = icare(A,B,Qcont,Rcont,0,E,0);
            obj.Kcont = Rcont\transpose(B)*S;
            
            % Determine Discrete LQR gain using Zero Order Hold ZOH
            discSysZOH = c2d(contSys,obj.thrustInterval,'zoh');
            AdZOH = discSysZOH.A;
            BdZOH = discSysZOH.B;
            SdZOH = idare(AdZOH,BdZOH,Qdisc,Rdisc,0,E);
            obj.KZOH = (Rdisc+transpose(BdZOH)*SdZOH*BdZOH)\transpose(BdZOH)*SdZOH*AdZOH;%;-R\transpose(BdZOH)*SdZOH;
            % Determine Discrete LQR using Impulse method
            discSysImp = c2d(contSys,obj.thrustInterval,'impulse');
            AdImp = discSysImp.A;
            BdImp = discSysImp.B;
            SdImp = idare(AdImp,BdImp,Qdisc,Rdisc,0,E);
            obj.Kimp = (Rdisc+transpose(BdImp)*SdImp*BdImp)\transpose(BdImp)*SdImp*AdImp;%Rdisc\transpose(BdImp)*SdImp;%
            
            % Give Formation Satellite 1 a Kalman filter
            if formationNo == 1  
                A4 = blkdiag(A,A,A,A);
                Aformation = blkdiag(A4,A4,A4);
                zerosB = zeros(size(B));
                Bformation = [-B    , B     , zerosB, zerosB, zerosB, zerosB;
                              -B    , zerosB, B     , zerosB, zerosB, zerosB;
                              -B    , zerosB, zerosB, B     , zerosB, zerosB;
                              -B    , zerosB, zerosB, zerosB, B     , zerosB;
                              zerosB,-B     , B     , zerosB, zerosB, zerosB;
                              zerosB, zerosB,-B     , B     , zerosB, zerosB;
                              zerosB, zerosB, zerosB, -B    , B     , zerosB;
                              zerosB, B     , zerosB, zerosB,-B     , zerosB;
                              zerosB, zerosB, zerosB, zerosB, B     ,-B;
                              zerosB, zerosB, zerosB, B     , zerosB,-B;
                              zerosB, zerosB, B     , zerosB, zerosB,-B;
                              zerosB, B     , zerosB, zerosB, zerosB,-B];
                Cformation =[eye(72);-eye(72)];
                G = eye(length(Aformation));
                H = [eye(length(Aformation));eye(length(Aformation))];
                kSys = ss(Aformation,[Bformation,G],Cformation,[[zeros(size(Bformation));zeros(size(Bformation))],H]);
                dSys = c2d(kSys,Constants.dt,'zoh');
                Q = blkdiag(1e-12*eye(3),1e-12*eye(3));
                
                Q = blkdiag(Q,Q,Q,Q,...
                            Q,Q,Q,Q,....
                            Q,Q,Q,Q);
               
                R = zeros(6,6);
                R(1:3,1:3) = Constants.posRelNoise^(2)*eye(3);
                R(4:6,4:6) = Constants.velRelNoise^(2)*eye(3);
                
                
                R = blkdiag(R,R,R,R,...
                            R,R,R,R,...
                            R,R,R,R,...
                            R,R,R,R,...
                            R,R,R,R,...
                            R,R,R,R);
                P = idare(dSys.A',dSys.C',Q,R,0,eye(size(dSys.A)));
                Man = P*dSys.C'/(dSys.C*P*dSys.C'+R);
                
                obj.P = P;
                obj.Kk = Man;
                obj.A = dSys.A;
                obj.B = dSys.B;
                obj.Bf = Bformation;
                obj.C = dSys.C;
                obj.D = dSys.D;
                obj.Q = Q;
                obj.Qinv= inv(dSys.C*P*dSys.C'+R);
                obj.extQ = Q;
                obj.R = R;
                
                obj.gk = zeros([36,1]);
                obj.faultThreshold = 25;
                obj.detect = 0 ;
                obj.isolate = 0;
                obj.faultDetTime = 0;
                obj.faultDetected = 0;
                % The temp matrix is used in the calculation of the
                % residual-fault transfer function and only calculated here
                % to save computation time
                obj.tempMatrix = obj.C/(eye(size(obj.A)) - obj.A + obj.A*obj.Kk*obj.C);
            end          
        end
        
        function updatePos(obj, stateChange)
            obj.position = obj.position + stateChange(1:3);
            obj.velocity = obj.velocity + stateChange(4:6);
        end
        
        function updateAtt(obj,attChange)
            obj.attitude = (obj.attitude + attChange(1:4))/(norm(obj.attitude + attChange(1:4)));
            obj.spin = obj.spin + attChange(5:7);
        end
            
        function state = getState(obj)
            state = [obj.position,obj.velocity];
        end
        
        function setState(obj,state)
            obj.position = state(1:3);
            obj.velocity = state(4:6);
        end
        
        % Navigation function
        % access actual position, add noise and bias
        function [posEst,velEst] = selfNav(obj)
            posEst = obj.position + obj.posSelfBias + obj.posSelfNoise*randn([1,3]);
            velEst = obj.velocity + obj.velSelfBias + obj.velSelfNoise*randn([1,3]);
        end
        
        % Relative navigation function
        % given another spacecraft, subtract positions, add noise and bias
        function relEst = relNav(obj)
            relDist1 = obj.spacecraftNeighbor1.position - obj.position ...
                       + obj.posRelBias + obj.posRelNoise * randn([1,3]);
            relVel1  = obj.spacecraftNeighbor1.velocity - obj.velocity ...
                       + obj.velRelBias + obj.velRelNoise * randn([1,3]);
            relDist2 = obj.spacecraftNeighbor2.position - obj.position ...
                       + obj.posRelBias + obj.posRelNoise * randn([1,3]);
            relVel2  = obj.spacecraftNeighbor2.velocity - obj.velocity ...
                       + obj.velRelBias + obj.velRelNoise * randn([1,3]);
            relDist3 = obj.spacecraftNeighbor3.position - obj.position ...
                       + obj.posRelBias + obj.posRelNoise * randn([1,3]);
            relVel3  = obj.spacecraftNeighbor3.velocity - obj.velocity ...
                       + obj.velRelBias + obj.velRelNoise * randn([1,3]);
            relDist4 = obj.spacecraftNeighbor4.position - obj.position ...
                       + obj.posRelBias + obj.posRelNoise * randn([1,3]);
            relVel4  = obj.spacecraftNeighbor4.velocity - obj.velocity ...
                       + obj.velRelBias + obj.velRelNoise * randn([1,3]);
            relEst = [relDist1,relVel1;
                      relDist2,relVel2;
                      relDist3,relVel3;
                      relDist4,relVel4;];
        end
        
        function relEst = relNavDistAngle(obj)
            qAtt = obj.attitude;
            
            rVec1 =  quatrotate(qAtt',(obj.spacecraftNeighbor1.position - obj.position));
            rVec2 =  quatrotate(qAtt',(obj.spacecraftNeighbor2.position - obj.position));
            rVec3 =  quatrotate(qAtt',(obj.spacecraftNeighbor3.position - obj.position));
            rVec4 =  quatrotate(qAtt',(obj.spacecraftNeighbor4.position - obj.position));

            r = euclidnorm(rVec1) + obj.rangeBias(1) + obj.rangeNoise * randn(1);
            theta = acos(rVec1(3)/euclidnorm(rVec1)) + obj.angleBias(1,1) +obj.angleNoise * randn(1);
            psi = atan2(rVec1(2),rVec1(1)) + obj.angleBias(1,2) + obj.angleNoise * randn(1);
            relDist1 = quatrotate(quatinv(qAtt'), r*[sin(theta)*cos(psi),sin(theta)*sin(psi),cos(theta)]);
                            
            relVel1  = obj.spacecraftNeighbor1.velocity - obj.velocity ...
                       + obj.velRelBias + obj.velRelNoise * randn([1,3]);
            
            r = euclidnorm(rVec2) + obj.rangeBias(2) + obj.rangeNoise * randn(1);
            theta = acos(rVec2(3)/euclidnorm(rVec2)) + obj.angleBias(2,1) + obj.angleNoise * randn(1);
            psi = atan2(rVec2(2),rVec2(1)) + obj.angleBias(2,2) + obj.angleNoise * randn(1);
            relDist2 = quatrotate(quatinv(qAtt'), r*[sin(theta)*cos(psi),sin(theta)*sin(psi),cos(theta)]);
            relVel2  = obj.spacecraftNeighbor2.velocity - obj.velocity ...
                       + obj.velRelBias + obj.velRelNoise * randn([1,3]);
            
            r = euclidnorm(rVec3) + obj.rangeBias(3) + obj.rangeNoise * randn(1);
            theta = acos(rVec3(3)/euclidnorm(rVec3)) + obj.angleBias(3,1) + obj.angleNoise * randn(1);
            psi = atan2(rVec3(2),rVec3(1)) + obj.angleBias(3,2) + obj.angleNoise * randn(1);
            relDist3 = quatrotate(quatinv(qAtt'), r*[sin(theta)*cos(psi),sin(theta)*sin(psi),cos(theta)]);
            relVel3  = obj.spacecraftNeighbor3.velocity - obj.velocity ...
                       + obj.velRelBias + obj.velRelNoise * randn([1,3]);
            
            r = euclidnorm(rVec4) + obj.rangeBias(4) + obj.rangeNoise * randn(1);
            theta = acos(rVec4(3)/euclidnorm(rVec4)) + obj.angleBias(4,1) + obj.angleNoise * randn(1);
            psi = atan2(rVec4(2),rVec4(1)) + obj.angleBias(4,2) + obj.angleNoise * randn(1);
            
            relDist4 = quatrotate(quatinv(qAtt'), r*[sin(theta)*cos(psi),sin(theta)*sin(psi),cos(theta)]);
            relVel4  = obj.spacecraftNeighbor4.velocity - obj.velocity ...
                       + obj.velRelBias + obj.velRelNoise * randn([1,3]);
            relEst = [relDist1,relVel1;
                      relDist2,relVel2;
                      relDist3,relVel3;
                      relDist4,relVel4;];
            
        end
        function navigation(obj,navType)
            obj.lastPositionEst = obj.positionEst;
            obj.lastVelocityEst = obj.velocityEst;
            obj.lastRelativeEst = obj.relativeEst;
            
            [pos,vel] = obj.selfNav();   
            obj.positionEst = pos;
            obj.velocityEst = vel;
            if navType == 1 
                obj.relativeEst = obj.relNav();
            elseif navType == 2 
                obj.relativeEst = obj.relNavDistAngle();
            end

            [a,e,inc,O,~,~,~,argLat,~] = rv2orb(pos',vel');
            n = sqrt(Constants.muEarth/(a^3*(1-e^2)^3))*(1+e*cos(argLat))^2;
            
            x = pos/norm(pos);
            z = cross(x,vel/norm(vel));
            y = cross(z,x)/(norm(cross(z,x)));
            obj.THillECI = [x;
                            y;
                            z];
            S = n*skewSym([0,0,1]);
            obj.Tdot = -S*obj.THillECI;%n*rotZdot(argLat)*rotX(inc)*rotZ(O);
          
            % Convert to Hill frame
            
            relPosHill = obj.THillECI*obj.relativeEst(:,1:3)';
            
            %relVelHill = THillECI*obj.relativeEst(:,4:6)'-S*THillECI*obj.relativeEst(:,1:3)';
            relVelHill = obj.THillECI*obj.relativeEst(:,4:6)'+obj.Tdot*obj.relativeEst(:,1:3)';
            posSelf = repmat(pos',[1,4]);
            velSelf = repmat(vel',[1,4]);
            posRel = posSelf+obj.relativeEst(:,1:3)';
            velRel = velSelf+obj.relativeEst(:,4:6)';
            obj.relEstHill = [relPosHill',relVelHill'];
        end
        % Kalman filter update
        function kalmanUpdate(obj,measurement,controlstates)
            % Project state into the present based on past state and control
            % input
            if ~ length(obj.kalmanEstimate) == 0
                estimatePriori = obj.A * obj.kalmanEstimate + obj.B(:,1:18) * obj.pastControl';%controlstates';
                % Propagate error covariance matrix
                obj.ErrCov = obj.A*obj.ErrCov*obj.A'+obj.Q;
                % Compute Kalman gain
                %K = obj.ErrCov*obj.C'/(obj.C*obj.ErrCov*obj.C'+obj.R);
                K = obj.Kk;
                % Update estimate using measurement
                obj.kalmanResidual =  (measurement'-(obj.C*estimatePriori +obj.D(:,1:18)* controlstates'));
                obj.kalmanEstimate = estimatePriori + K * obj.kalmanResidual;
                % Update Error Covariance Matrix
                I = eye(size(obj.A));
                obj.ErrCov = (I - K * obj.C)*obj.ErrCov*(I -K * obj.C)' + K*obj.R*K';
                % Low Pass Filter
                obj.kalLowPass = obj.alphLow * obj.kalmanResidual' + (1-obj.alphLow) * obj.kalLowPass;
                obj.pastControl = controlstates;
            else
                obj.kalmanEstimate = 0.5*(measurement(1,1:72)-measurement(1,73:end))';
                obj.kalmanResidual = zeros(size(measurement))';
                obj.kalLowPass = zeros(size(measurement));
                obj.pastControl = controlstates;
                obj.ErrCov = obj.Q;
            end
        end
        
        % Extended Kalman filter update DEPRECATED DONT USE
        function extendedKalmanUpdate(obj,time,measurement,controlstates)
            % Project state into future based on past state and control
            % input
            state = obj.lastRelativeEst;
            satellitePosECI = obj.lastPositionEst;
            satelliteVelECI = obj.lastVelocityEst;
            posCenterECI = satellitePosECI + mean(state(:,1:3),1);
            velCenterECI = satelliteVelECI + mean(state(:,4:6),1);
            [a,e,~,~,~,~,~,argLat,~] = rv2orb(posCenterECI',velCenterECI');
            
            if ~ length(obj.extKalmanEstimate) == 0
                % Propagate State with nonlinear function
                Alin =  expm(dynJac(obj.extKalmanEstimate,posCenterECI,velCenterECI,a,e,argLat)*Constants.dt);

                estimatePriori = Alin*obj.extKalmanEstimate + obj.B(1:72,1:18)*controlstates';
                % Linearly approximate function
                % Jacobian 
                % Propagate error covariance matrix
                obj.extErrCov = Alin*obj.extErrCov*Alin'+obj.Q;
                % Compute Kalman gain
                K = obj.extErrCov*obj.C'/(obj.C*obj.extErrCov*obj.C'+obj.R);
                %K = obj.Kk;
                % Update estimate using measurement
                observation =  obj.C*estimatePriori;
                obj.extKalmanResidual = (measurement'- observation);
                obj.extKalmanEstimate = (estimatePriori +  K * obj.extKalmanResidual);
                % Update Error Covariance Matrix
                I = eye(size(Alin));
                obj.extErrCov = (I - K * obj.C)*obj.extErrCov*(I - K *obj.C)' + K*obj.R*K';
            else
                obj.extKalmanEstimate = 0.5*(measurement(1,1:72)-measurement(1,73:end))';
                obj.extKalmanResidual = zeros(size(measurement));
                obj.extErrCov = obj.Q;
            end
        end
        
        function fdi(obj,time)
            % Perform a recursive formulation of a GLR test for various
            % 6*n fault scenarios
            residual = obj.kalmanResidual';%-obj.kalLowPass;
            I = eye(72);
            TECIBody =myQuat2dcm(quatinv(obj.formationOrientation));
            TConf2Acc = 1/obj.mass*obj.THillECI*TECIBody*obj.Tconfig;
            TConfStack = blkdiag(TConf2Acc,TConf2Acc,TConf2Acc,...
                                 TConf2Acc,TConf2Acc,TConf2Acc);
            Ff = obj.B(:,1:18)*TConfStack;

            Du = obj.D(:,1:18)*TConfStack;
            K = obj.Kk;
           
            Mat = -obj.tempMatrix*(Ff-obj.A*K*Du)-Du;
            nf = 6*6;
             
            for ii = 1:nf
                fVec = zeros([nf,1]);
                fVec(ii) = 0.0005;
                mu = Mat*fVec;
                sz = (mu-obj.residualMean')'*obj.Qinv*(residual'-1/2*(mu+obj.residualMean'));

                obj.gk(ii) = max(0,obj.gk(ii) + sz);
            end
            if any(obj.gk>obj.faultThreshold)
                obj.detect = 1;
                [~,obj.isolate] = max(obj.gk);
                obj.isoCounter = obj.isoCounter + 1;
                if ~obj.faultDetected
                    obj.faultDetTime = time;
                    obj.faultDetected = true;
                end

            end 
        end
        
        % Guidance Law
        % determine position to be in for current time/trajectory to follow
        function deltaV = guidance(obj,time)
            if mod(time,obj.thrustInterval) == 0    
                debugGuidance = 1;
                if debugGuidance
                    a       = obj.formationCenterOrbit(1);
                    e       = obj.formationCenterOrbit(2);
                    inc     = obj.formationCenterOrbit(3);
                    O       = obj.formationCenterOrbit(4);
                    o       = obj.formationCenterOrbit(5);
                    nu      = obj.formationCenterOrbit(6);
                    truLon  = obj.formationCenterOrbit(7);
                    argLat  = obj.formationCenterOrbit(8);
                    lonPer  = obj.formationCenterOrbit(9);
                    p       = obj.formationCenterOrbit(10);
                    [pos,~] = keplerEQsolve(a,e,inc,O,o,nu,truLon,argLat,lonPer,p,time+obj.thrustInterval);
                else
                    %Test propagation using virtual center 
                    state = obj.relNav();
                    stateR = reshape(state,6,4);
                    center = obj.selfNav()' + mean(stateR(1:3,:),2);
                    inc = obj.formationCenterOrbit(3);
                    velMag = sqrt(Constants.muEarth/norm(center));
                    centerDir = center/norm(center);
                    nDir = [0,-sin(inc),cos(inc)];
                    velDir = cross(nDir,centerDir);
                    velCenter = velMag*velDir;
                    [a,e,inc,O,o,nu,truLon,argLat,lonPer,p] = rv2orb(center,velCenter');
                    [pos,~] = keplerEQsolve(a,e,inc,O,o,nu,truLon,argLat,lonPer,p,obj.thrustInterval);
                end
                
                posNext = pos+obj.centerOffset;
                [v1B,v2B] = lambertBook(obj.selfNav(),posNext,obj.thrustInterval,'pro');
                deltaV = v1B-obj.velocity;
                obj.accumDV = obj.accumDV +norm(deltaV);
                obj.velocity = v1B;
            end
            
        end
        
        % Control Law
        % Determine control impulse based on estimated position and
        % position/trajectory 
        function controlCommand(obj, time,controltype,disctype)
            % Check if we have a form of discrete control (disctype =|= 0)
            if disctype ~= 0
                % Check if we are outside a burn window
                if mod(time,obj.thrustInterval)>obj.burnTime
                    obj.cImpulse = [0;0;0];
                    obj.thrustAllocComplete = 0;
                    return
                elseif any(obj.cImpulse)% Check if we already were in this burn window
                    return
                end
            end
                    
            switch controltype
                case 1
                    err =  obj.errTrackedCenter(time);
                case 2
                    err =  obj.errVirtualCenter(time);
            end
            
            % Select control gain according to discretization method
            % 0     =     Continuous
            % 1     =     Zero Order Hold (ZOH)
            % 2     =     Impulse (imp)
            switch disctype
                case 0 
                    K = obj.Kcont;
                case 1
                    K = obj.KZOH;
                case 2 
                    K = obj.Kimp;
            end

            cImpulse = -K*err';
            obj.cImpulseHill = cImpulse;
            
            % Transform control force to ECI frame
            state = obj.relativeEst;
            %stateR = reshape(state,6,4);
            satellitePosECI = obj.positionEst;
            velECI = obj.velocityEst; 
            posCenterECI = satellitePosECI + mean(state(:,1:3),1);
            % Estimate center velocity
            velCenterECI = velECI + mean(state(:,4:6),1);
            % Transform error into Hill Frame
            
            % INCORRECT TRANSFORMATION, DEPRECATED only used for continutiy
            x = velCenterECI/norm(velCenterECI);
            z = posCenterECI/norm(posCenterECI);
            y = cross(z,x)/(norm(cross(z,x)));
            THillECI_Incorrect = [x;
                            y;
                            z];
            cImpulse = THillECI_Incorrect\cImpulse;            
            %cImpulse = obj.THillECI\cImpulse;
            
            if disctype>0
                cImpulse = cImpulse*obj.thrustInterval*0.8;
            end
            obj.cImpulse = cImpulse;
            obj.accumDV = obj.accumDV + norm(cImpulse);
        end

        % Thruster allocation
        % Given control force vector, determine which thrusters should fire
        % for how long
        function thrustAlloc(obj,time)
            if any(obj.cImpulse) && ~obj.thrustAllocComplete
                obj.thrustAllocComplete = 1;
                impulseECI = obj.cImpulse;
                % Determine current attitude
                qAtt = obj.attitude();
                % Transform impulse into body frame
                impulseBody = quatrotate(qAtt',impulseECI');
                % Calculate opening times
                opening = 1/obj.thrust*obj.TconfigInv*obj.mass*impulseBody';
                % Negative opening times are added to other thruster
                if opening(1)<0
                    opening(2) = opening(2)-opening(1);
                    opening(1) = 0;
                end
                if opening(2)<0
                    opening(1) = opening(1)-opening(2);
                    opening(2) = 0;
                end
                if opening(3)<0
                    opening(4) = opening(4)-opening(3);
                    opening(3) = 0;
                end
                if opening(4)<0
                    opening(3) = opening(3)-opening(4);
                    opening(4) = 0;
                end
                if opening(5)<0
                    opening(6) = opening(6)-opening(5);
                    opening(5) = 0;
                end
                if opening(6)<0
                    opening(5) = opening(5)-opening(6);
                    opening(6) = 0;
                end      
                obj.thrusterOpeningCount(opening>0) = obj.thrusterOpeningCount(opening>0) + 1;
                obj.thrusterOpeningTime = obj.thrusterOpeningTime + opening;
                obj.spentProp = obj.spentProp + sum(opening)*obj.thrust/(obj.isp*9.81);
                obj.opTimes = time+opening;
                obj.maxBurnTime = max(obj.maxBurnTime,max(opening));
            end
        end
        
        % Determine force based on opening times
        function cFECI = controlForce(obj,time)
            remainingBurntime = max(obj.opTimes-time,0);
            minOpTime = obj.minImpulse/obj.thrust;
            for ii = 1:6
                if remainingBurntime(ii) > 0
                    if remainingBurntime(ii) < minOpTime
                        remainingBurntime(ii) = minOpTime;
                    end
                end
            end
            thrusterOpening = remainingBurntime>0;  
            burnFraction = min(abs(remainingBurntime)/Constants.dt,1);
            obj.burnFraction = burnFraction;
            if time > obj.faultTime
                % Determine 
                thrusterForce = max(obj.thrust*obj.faultVectorClosed.*burnFraction.*thrusterOpening,...
                                obj.thrust*obj.faultVectorOpen);
                if any(1-obj.faultVectorClosed)
                    faultyThrusterActivation = find(1-obj.faultVectorClosed) == find(thrusterOpening);

                    if any(faultyThrusterActivation) && ~ obj.firstThrust
                        obj.faultTime = time;
                        obj.firstThrust = true;
                    end
                end
            else
                thrusterForce = obj.thrust*burnFraction.*thrusterOpening;                               
            end
            cFBody = obj.Tconfig*thrusterForce;
            cFECI = quatrotate(quatinv(obj.attitude'),cFBody');
            obj.cFECI = cFECI;
            % For Kalman filter purposes, get commanded force output in
            % Hill frame
            thrusterForceFaultless = obj.thrust*burnFraction.*thrusterOpening;
            cFfaultless = obj.Tconfig*thrusterForceFaultless;
            cFECIfaultless = quatrotate(quatinv(obj.attitude'),cFfaultless');
            obj.cAccHill = obj.THillECI * cFECIfaultless'/obj.mass;
            
            deltaMass = sum(thrusterForce)/(obj.isp*9.81);
            obj.mass = obj.mass-deltaMass;
        end
        %
        function err = errVirtualCenter(obj,time) 
            
            state = obj.relativeEst;
            satellitePosECI = obj.positionEst;
            velECI = obj.velocityEst; 
            posCenterECI = satellitePosECI + mean(state(:,1:3),1);
            % Estimate center velocity
            velCenterECI = velECI + mean(state(:,4:6),1);
            [a,~,inc,O,~,theta,~,argLat,~] = rv2orb(posCenterECI',velCenterECI');
            n = sqrt(Constants.muEarth/a^3);
            realCenterECI = satellitePosECI + obj.centerOffset;
            % Transform error into Hill Frame
            % NOTE: INCORRECT TRANSFORMATION, DEPRECATED, used only for
            % consistency with Data generation method. CORRECT
            % transformations are given by obj.THillECI and obj.Tdot
            x = velCenterECI/norm(velCenterECI);
            z = posCenterECI/norm(posCenterECI);
            y = cross(z,x)/(norm(cross(z,x)));
            THillECI_Incorrect = [x;
                            y;
                            z];
            Tdot_Incorrect = n*rotX(0.5*pi)*rotZdot(theta)*rotZ(0.5*pi)*rotX(inc)*rotZ(O);
          
            vHill = THillECI_Incorrect*(velECI-velCenterECI)'-Tdot_Incorrect*THillECI_Incorrect*(satellitePosECI-posCenterECI)';
            %vHill = obj.THillECI*(velECI-velCenterECI)'+obj.Tdot*(satellitePosECI-posCenterECI)';
            
            vRef = Tdot_Incorrect*THillECI_Incorrect*obj.centerOffset';
            %vRef = -obj.Tdot*obj.centerOffset';
            err(1:3) = THillECI_Incorrect*(realCenterECI - posCenterECI)';
            %err(1:3) = obj.THillECI*(realCenterECI - posCenterECI)';
            err(4:6) = vHill-vRef;

            obj.cErr = err;
        end
        
        function err= errTrackedCenter(obj,time)
            a       = obj.formationCenterOrbit(1);
            e       = obj.formationCenterOrbit(2);
            inc     = obj.formationCenterOrbit(3);
            O       = obj.formationCenterOrbit(4);
            o       = obj.formationCenterOrbit(5);
            nu      = obj.formationCenterOrbit(6);
            truLon  = obj.formationCenterOrbit(7);
            argLat  = obj.formationCenterOrbit(8);
            lonPer  = obj.formationCenterOrbit(9);
            p       = obj.formationCenterOrbit(10);
            [posCenterECI,velCenterECI,theta] = keplerEQsolve(a,e,inc,O,o,nu,truLon,argLat,lonPer,p,time);
            satellitePosECI = obj.positionEst;
            velECI = obj.velocityEst;
            n = sqrt(Constants.muEarth/a^3);
            %{
            state = obj.relNav();
            stateR = reshape(state,6,4);
            realCenterECI = obj.position + mean(stateR(1:3,:),2)';
            %}
            realCenterECI = satellitePosECI + obj.centerOffset;
            inc = obj.formationCenterOrbit(3);
            % Transform error into Hill Frame
            
            vHill = obj.THillECI*(velECI-velCenterECI)'-obj.Tdot*obj.THillECI*(satellitePosECI-posCenterECI)';
            vRef = obj.Tdot*obj.THillECI*obj.centerOffset';
            err(1:3) = obj.THillECI*(realCenterECI - posCenterECI)';
            err(4:6) = vHill-vRef;      
            
        end
    end
end
