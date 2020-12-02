function angle = angleVec(vec1,vec2)
%ANGLEVEC Determine angle between two given vectors
%   Detailed explanation goes here
angle = atan2(norm(cross(vec1,vec2)),dot(vec1,vec2));

end

