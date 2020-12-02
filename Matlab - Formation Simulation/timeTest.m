
%% Setup
seed = 6;
rng(seed)
%Kalman Setup
J2 = 0.0010826357;
r = 7.0656e+06;
n = sqrt(Constants.muEarth/r^3);
inc = 62.8110/360*2*pi;
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
P = P;
Kk = Man;
A = dSys.A;
B = dSys.B;
Bf = Bformation;
C = dSys.C;
D = dSys.D;
Q = Q;
Qinv= inv(dSys.C*P*dSys.C'+R);
extQ = Q;
R = R;
                
gk = zeros([36,1]);
faultThreshold = 25;

residualMean = rand(1,24*6);
kalmanResidual = rand(1,24*6);

THillECI = rand(3,3);
Tconfig  = [1,-1,0,0 ,0,0 ;
           0,0 ,1,-1,0,0 ;
           0,0 ,0,0 ,1,-1];% thruster configuration matrix
TconfigInv = [0.5,0   ,0   ;
             -0.5,0   ,0   ;
              0  ,0.5 ,0   ;
              0  ,-0.5,0   ;
              0  ,0   ,0.5 ;
              0  ,0   ,-0.5]
formationOrientation = rand(1,4);
formationOrientation = formationOrientation/norm(formationOrientation);
tempMatrix = C/(eye(size(A)) - A + A*Kk*C);

%% Neural Nets
netInput = rand(24,1);
% Detection
numDetUnits = 128;
detNet = [...
       sequenceInputLayer(24)
       lstmLayer(numDetUnits)
       lstmLayer(numDetUnits)
       fullyConnectedLayer(64)
       tanhLayer];
numIsoUnits = 128;
          
% Isolation

isoLayers = [...
       sequenceInputLayer(24)
       lstmLayer(256)
       lstmLayer(256)
       fullyConnectedLayer(100)
       softmaxLayer];
isoNet = network(isoLayers);
isoNet = init(isoNet)

%% Kalman
tic
residual = kalmanResidual';%-obj.kalLowPass;
I = eye(72);

TECIBody =myQuat2dcm(quatinv(formationOrientation));
TConf2Acc = 1/mass*THillECI*TECIBody*Tconfig;
TConfStack = blkdiag(TConf2Acc,TConf2Acc,TConf2Acc,...
                     TConf2Acc,TConf2Acc,TConf2Acc);
Ff = B(:,1:18)*TConfStack;

Du = D(:,1:18)*TConfStack;

Mat = -tempMatrix*(Ff-A*Kk*Du)-Du;
nf = 6*6;
for ii = 1:nf
    fVec = zeros([nf,1]);
    fVec(ii) = 0.0005;
    mu = Mat*fVec;
    sz = (mu-residualMean')'*Qinv*(residual-1/2*(mu+residualMean'));
    gk(ii) = max(0,gk(ii) + sz);
end

if any(gk>faultThreshold)
    detect = 1;
    [~,obj.isolate] = max(obj.gk);
    isoCounter = obj.isoCounter + 1;
    if ~faultDetected
        faultDetTime = time;
        faultDetected = true;
    end
end 
toc