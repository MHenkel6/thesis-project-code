function devTotal = relativeDynNonlinear(formationState,formCenter,velCenter,a,e,argLat)
%Integrate State over 1 second using non-linear dynamics
devTotal = zeros(size(formationState));



dt = Constants.dt;
%s = reshape(formationState,[12,6]);
for ii = 1:12
    x = formationState((ii-1)*6+1:ii*6);
    %{
    if ii<5
        satCenter = formCenter + (s(1:3,1)+s(1:3,2)+s(1:3,3)+s(1:3,4))/4;
    elseif ii == 5
        satCenter = formCenter + (-s(1:3,1)+s(1:3,5)-s(1:3,8)-s(1:3,12))/4;
    elseif ii == 6
        satCenter = formCenter + (-s(1:3,2)-s(1:3,5)+s(1:3,6)-s(1:3,11))/4;
    elseif ii == 7
        satCenter = formCenter + (-s(1:3,3)-s(1:3,6)+s(1:3,7)-s(1:3,10))/4;
    elseif ii == 8
        satCenter = formCenter + (-s(1:3,4)-s(1:3,7)+s(1:3,8)-s(1:3,9))/4;
    else
        satCenter = formCenter + (s(1:3,9)+s(1:3,10)+s(1:3,11)+s(1:3,12))/4;
    end
    %}
    dev1 =  nonLinRel(x,formCenter,velCenter,a,e,argLat);
    dev2 =  nonLinRel(x + dev1*dt/2,formCenter,velCenter,a,e,argLat);
    dev3 =  nonLinRel(x + dev2*dt/2,formCenter,velCenter,a,e,argLat);
    dev4 =  nonLinRel(x + dev3*dt,formCenter,velCenter,a,e,argLat);
    devTotal((ii-1)*6+1:ii*6) = dt*(dev1 + 2*dev2 + 2*dev3 + dev4)/6;
    devTotal((ii-1)*6+1:ii*6) = dev1;
end

end

