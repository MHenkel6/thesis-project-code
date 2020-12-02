function mJ = measJac(state,h)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
mJ = zeros(144,length(state));
defMeas = relativeMeasNonlinear(state);
for ii = 1:length(state)
    hVec = zeros(size(state));
    hVec(ii) = h;
    mJ(:,ii) = (relativeMeasNonlinear(state+hVec)-defMeas)/h;
end
end

