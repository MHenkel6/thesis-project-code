function [ dcm ] = myQuat2dcm( qin )
%myQuat2dcm Customized quaternion to direction cosine matri


dcm = zeros(3,3);
a = qin(:,1).^2;
b = qin(:,2).^2;
c = qin(:,3).^2;
d = qin(:,4).^2;

e = qin(:,2).*qin(:,3);
f = qin(:,1).*qin(:,4);
g = qin(:,2).*qin(:,4);
h = qin(:,1).*qin(:,3);
k = qin(:,3).*qin(:,4);
l = qin(:,1).*qin(:,2);

dcm(1,1) = a + b - c - d;
dcm(1,2) = 2.*(e + f);
dcm(1,3) = 2.*(g - h);
dcm(2,1) = 2.*(e - f);
dcm(2,2) = a - b + c - d;
dcm(2,3) = 2.*(k + l);
dcm(3,1) = 2.*(g + h);
dcm(3,2) = 2.*(k - l);
dcm(3,3) = a - b - c + d;
end

