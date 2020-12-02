function [ qout ] = myQuatRotate( q,r )
%myQuatRotate Updated quaternion rotate

dcm = myQuat2dcm(q);

if ( size(q,1) == 1 ) 
    % Q is 1-by-4
    qout = (dcm*r')';
elseif (size(r,1) == 1) 
    % R is 1-by-3
    for i = size(q,1):-1:1
        qout(i,:) = (dcm(:,:,i)*r')';
    end
else
    % Q is M-by-4 and R is M-by-3
    for i = size(q,1):-1:1
        qout(i,:) = (dcm(:,:,i)*r(i,:)')';
    end


end

